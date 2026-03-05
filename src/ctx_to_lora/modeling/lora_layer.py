from collections.abc import Iterable
from functools import partial
from operator import attrgetter

import torch
import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float, Integer
from torch import Tensor

from ctx_to_lora.utils import get_layers


def lora_forward(
    x: Float[Tensor, "tot_q seq_len d_in"],
    n_qs: Integer[Tensor, "n_ctx"],
    tot_q: int,
    A: Float[Tensor, "n_ctx r d_in"],
    B: Float[Tensor, "n_ctx r d_out"],
    lora_dropout_p: float,
    scaling: float,
    self,
    *args,
    **kwargs,
) -> Float[Tensor, "tot_q seq_len d_out"]:
    # A: [n_ctx, r, d_in] -> [tot_q, r, d_in]
    A = A.repeat_interleave(n_qs, dim=0, output_size=tot_q)
    # B: [n_ctx, d_out, r] -> [tot_q, d_out, r]
    B = B.repeat_interleave(n_qs, dim=0, output_size=tot_q)

    # Use base_layer(x) instead of torch.nn.Linear.forward(self, x) to support
    # quantized linear layers (e.g. MXFP4) that override forward for dequantization.
    base_out = self.base_layer(x, *args, **kwargs)
    x = x.to(A.dtype)
    delta_x = F.dropout(x, p=lora_dropout_p, training=self.training)
    delta_x = einsum(A, delta_x, "tot_q r d_in, tot_q s_len d_in -> tot_q s_len r")
    delta_x = einsum(B, delta_x, "tot_q r d_out, tot_q s_len r -> tot_q s_len d_out")
    delta_x = delta_x * scaling
    return (base_out + delta_x).to(base_out.dtype)


def lora_forward_packed(
    x: Float[Tensor, "1 tot_len d_in"],
    n_qs: Integer[Tensor, "n_ctx"],
    tot_q: int,
    seq_lens: Integer[Tensor, "tot_q"],
    tot_len: int,
    A: Float[Tensor, "n_ctx r d_in"],
    B: Float[Tensor, "n_ctx r d_out"],
    lora_dropout_p: float,
    scaling: float,
    self,
    *args,
    **kwargs,
) -> Float[Tensor, "1 tot_len d_out"]:
    # bs of x should be 1 in this case
    # Use base_layer(x) instead of torch.nn.Linear.forward(self, x) to support
    # quantized linear layers (e.g. MXFP4) that override forward for dequantization.
    base_out = self.base_layer(x, *args, **kwargs)
    x = x.to(A.dtype)
    delta_x = F.dropout(x, p=lora_dropout_p, training=self.training)
    repeated_A = A.repeat_interleave(n_qs, dim=0, output_size=tot_q)
    repeated_A = repeated_A.repeat_interleave(seq_lens, dim=0, output_size=tot_len)

    repeated_B = B.repeat_interleave(n_qs, dim=0, output_size=tot_q)
    repeated_B = repeated_B.repeat_interleave(seq_lens, dim=0, output_size=tot_len)

    delta_x = einsum(
        repeated_A, delta_x, "tot_len r d_in, bs tot_len d_in -> bs tot_len r"
    )
    delta_x = einsum(
        repeated_B, delta_x, "tot_len r d_out, bs tot_len r -> bs tot_len d_out"
    )
    delta_x = delta_x * scaling

    return (base_out + delta_x).to(base_out.dtype)


def moe_lora_forward_packed(
    hidden_states,
    router_indices,
    routing_weights,
    moe_lora_params=None,
    self_module=None,
    lora_dropout_p=0.0,
    scaling=1.0,
    moe_lora_strategy="shared",
):
    """Replaces GptOssExperts.forward, adding LoRA delta after down_proj for fired experts only.

    Strategies:
        shared: full rank R applied to each fired expert (same A, B for all)
        split:  rank R/top_k per fired expert, sliced by routing rank position
    """
    if moe_lora_params is None:
        return self_module.forward_orig(hidden_states, router_indices, routing_weights)

    batch_size = hidden_states.shape[0]
    hs_flat = hidden_states.reshape(-1, self_module.hidden_size)
    num_experts = self_module.gate_up_proj.shape[0]
    top_k = router_indices.shape[1]
    next_states = torch.zeros_like(hs_flat)

    # For split strategy: precompute per-token rank position of each expert
    # router_indices: [num_tokens, top_k] — position 0 = highest-ranked expert
    if moe_lora_strategy == "split":
        # Build a map: expert_rank_pos[token_idx, expert_idx] = rank position (0..top_k-1)
        # Only defined for fired experts; we look it up per expert in the loop
        expert_rank_pos = torch.full(
            (hs_flat.shape[0], num_experts), -1,
            dtype=torch.long, device=hidden_states.device,
        )
        for k in range(top_k):
            expert_rank_pos.scatter_(1, router_indices[:, k:k+1], k)

    with torch.no_grad():
        expert_mask = F.one_hot(router_indices, num_classes=num_experts + 1).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit[:]:
        expert_idx = expert_idx[0]
        if expert_idx == num_experts:
            continue
        with torch.no_grad():
            _, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hs_flat[token_idx]

        gate_up = current_state @ self_module.gate_up_proj[expert_idx] + self_module.gate_up_proj_bias[expert_idx]
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self_module.limit)
        up = up.clamp(min=-self_module.limit, max=self_module.limit)
        glu = gate * torch.sigmoid(gate * self_module.alpha)
        gated_output = (up + 1) * glu

        out = gated_output @ self_module.down_proj[expert_idx] + self_module.down_proj_bias[expert_idx]

        # === LoRA DELTA (fired experts only) ===
        if "down_proj" in moe_lora_params:
            A = moe_lora_params["down_proj"]["A"][token_idx]  # [n_tokens, r, d_in]
            B = moe_lora_params["down_proj"]["B"][token_idx]  # [n_tokens, r, d_out]

            if moe_lora_strategy == "split":
                # Slice rank dimension by this expert's routing rank position
                r_per_k = A.shape[1] // top_k
                rank_pos = expert_rank_pos[token_idx, expert_idx]  # [n_tokens]
                # Gather the correct rank slice per token
                r_start = rank_pos * r_per_k  # [n_tokens]
                # Build index for gathering: [n_tokens, r_per_k]
                r_idx = r_start.unsqueeze(1) + torch.arange(r_per_k, device=A.device)
                A = A.gather(1, r_idx.unsqueeze(2).expand(-1, -1, A.shape[2]))
                B = B.gather(1, r_idx.unsqueeze(2).expand(-1, -1, B.shape[2]))

            gated_f = F.dropout(gated_output.to(A.dtype), p=lora_dropout_p, training=self_module.training)
            delta = einsum(A, gated_f, "n r d_in, n d_in -> n r")
            delta = einsum(B, delta, "n r d_out, n r -> n d_out")
            out = out + delta.to(out.dtype) * scaling

        # routing_weights may be [num_tokens, num_experts] or [num_tokens, top_k]
        if routing_weights.shape[1] == num_experts:
            expert_weight = routing_weights[token_idx, expert_idx, None]
        else:
            top_k_mask = (router_indices[token_idx] == expert_idx)
            expert_weight = (routing_weights[token_idx] * top_k_mask).sum(dim=-1, keepdim=True)
        weighted_output = out * expert_weight
        next_states.index_add_(0, token_idx, weighted_output.to(hs_flat.dtype))

    return next_states.view(batch_size, -1, self_module.hidden_size)


def apply_lora_to_layers(
    model: torch.nn.Module,
    layer_indices: Iterable[int],
    generated_loras: dict[str, dict[str, Float[Tensor, "n_ctx n_layers r _"]]],
    n_qs: Integer[Tensor, "n_ctx"],
    position_ids: Integer[Tensor, "bs seq_len"] = None,
    moe_target_modules: list[str] | None = None,
) -> None:
    layers = get_layers(model)
    if position_ids is not None:
        position_ids = position_ids.squeeze(0)
        seq_lens = position_ids[torch.where(position_ids == 0)[0][1:] - 1]
        seq_lens = torch.cat(
            [seq_lens, torch.tensor([position_ids[-1]], device=seq_lens.device)]
        )
        seq_lens += 1
        tot_len = seq_lens.sum().item()
    tot_q = n_qs.sum().item()
    moe_targets = set(moe_target_modules or [])

    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        moe_lora_params = {}

        for mname in generated_loras:
            A = generated_loras[mname]["A"][:, layer_idx]
            B = generated_loras[mname]["B"][:, layer_idx]

            if mname in moe_targets:
                # Pre-expand to per-token for MoE expert forward
                A_exp = A.repeat_interleave(n_qs, dim=0, output_size=tot_q)
                B_exp = B.repeat_interleave(n_qs, dim=0, output_size=tot_q)
                if position_ids is not None:
                    A_exp = A_exp.repeat_interleave(seq_lens, dim=0, output_size=tot_len)
                    B_exp = B_exp.repeat_interleave(seq_lens, dim=0, output_size=tot_len)
                moe_lora_params[mname] = {"A": A_exp, "B": B_exp}
            else:
                # Existing PEFT path
                if mname in ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"]:
                    long_mname = f"self_attn.{mname}"
                elif mname in ["down_proj", "up_proj", "gate_proj"]:
                    long_mname = f"mlp.{mname}"
                module = attrgetter(long_mname)(layer)
                module.forward = partial(module.forward, n_qs=n_qs, tot_q=tot_q, A=A, B=B)
                if position_ids is not None:
                    module.forward = partial(
                        module.forward, seq_lens=seq_lens, tot_len=tot_len
                    )

        if moe_lora_params:
            experts = layer.mlp.experts
            experts.forward = partial(experts.forward, moe_lora_params=moe_lora_params)
