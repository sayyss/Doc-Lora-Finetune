"""Quick gradient check — verify hypernet params get non-zero gradients."""
import os, torch
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from ctx_to_lora.configs import ArgumentParser, DataArguments, CtxTrainingArguments, ModelArguments, LoRAArguments, TrainingArguments, HypernetArguments, AggregatorArguments, CtxEncoderArguments, ExperimentSetup
from ctx_to_lora.model_loading import get_model_and_tokenizer, get_lora_config, get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel, get_hypernet_config
from transformers import AutoConfig, set_seed

set_seed(42)

model_name = "openai/gpt-oss-20b"
base_model, tokenizer = get_model_and_tokenizer(
    model_name_or_path=model_name, train=True, requires_grad=False,
    peft_config=get_lora_config(model_name, target_modules=["q_proj"], lora_r=8, lora_dropout=0.0),
    use_flash_attn=True,
)

moe_target_modules = ["down_proj"]
ctx_encoder_model_config = base_model.config
hypernet_config = get_hypernet_config(
    base_model, ctx_encoder_model_config,
    type("H", (), {"per_rank_gen": True, "per_layer_processing": True, "num_pre_head_layers": 4})(),
    type("A", (), {"n_latent_queries": 8, "num_blocks": 9, "num_self_attn_per_block": 0})(),
    type("C", (), {"ctx_encoder_type": "per_layer_activations", "ctx_encoder_model_name_or_path": None, "layer_idx": None, "ctx_encoder_last_layer": None})(),
    moe_target_modules=moe_target_modules,
    moe_lora_strategy="shared",
)

ctx_encoder_args = type("CE", (), {
    "ctx_encoder_model_name_or_path": None, "layer_idx": 6, "ctx_encoder_last_layer": None,
    "ctx_encoder_type": "per_layer_activations",
})()

model = ModulatedPretrainedModel(base_model, hypernet_config, ctx_encoder_args)
model.train()

# DO NOT compile — we want to check raw gradient flow
print("=== Skipping torch.compile for gradient check ===")

# Create a minimal input
ctx_ids = torch.randint(0, 1000, (1, 64), device=model.device)
ctx_attn_mask = torch.ones_like(ctx_ids)
input_ids = torch.randint(0, 1000, (1, 32), device=model.device)
labels = input_ids.clone()
labels[:, :16] = -100
position_ids = torch.arange(32, device=model.device).unsqueeze(0)
n_ctx_chunks = torch.tensor([1], device=model.device)
n_queries = torch.tensor([1], device=model.device)

outputs = model(
    ctx_ids=ctx_ids, ctx_attn_mask=ctx_attn_mask,
    n_ctx_chunks=n_ctx_chunks, n_queries=n_queries,
    input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
    position_ids=position_ids, labels=labels,
)

loss = outputs.loss
print(f"Loss: {loss.item()}")
loss.backward()

# Check gradients on hypernet parameters
total_params = 0
nonzero_grad_params = 0
zero_grad_params = 0
none_grad_params = 0

for name, param in model.hypernet.named_parameters():
    if param.requires_grad:
        total_params += 1
        if param.grad is None:
            none_grad_params += 1
            if total_params <= 5:
                print(f"  NONE grad: {name}")
        elif param.grad.abs().sum() == 0:
            zero_grad_params += 1
            if total_params <= 5:
                print(f"  ZERO grad: {name}")
        else:
            nonzero_grad_params += 1
            if nonzero_grad_params <= 5:
                print(f"  OK grad: {name} grad_norm={param.grad.norm().item():.6f}")

print(f"\n=== GRADIENT CHECK ===")
print(f"Total trainable params: {total_params}")
print(f"Non-zero gradients: {nonzero_grad_params}")
print(f"Zero gradients: {zero_grad_params}")
print(f"None gradients: {none_grad_params}")

if nonzero_grad_params > 0:
    print("\nGRADIENTS ARE FLOWING — grad_norm=0 in training is a torch.compile artifact")
else:
    print("\nGRADIENTS ARE NOT FLOWING — there is a real issue")
