# GPT-OSS 20B D2L Training Fixes

Documented fixes to train a Doc-to-LoRA hypernetwork targeting `openai/gpt-oss-20b` (21B MoE, MXFP4 quantized).

## Model Background

GPT-OSS 20B is a Mixture-of-Experts model (32 experts, 4 active, 3.6B active params, 20.91B total). Key constraints:
- Expert weights are batched `nn.Parameter` (not `nn.Linear`) -- LoRA can only target attention modules
- Ships in MXFP4 quantization -- must be dequantized to bf16 for training
- Uses `vllm-flash-attn3` attention by default -- inference-only, breaks training
- Chat template uses `<|start|>`, `<|channel|>`, `<|message|>`, `<|end|>`, `<|return|>` tokens

---

## Fix 1: MXFP4 Dequantization to bf16

**File:** `src/ctx_to_lora/model_loading.py`

**Problem:** GPT-OSS 20B is stored in MXFP4 format. MXFP4 is an inference-only format that doesn't support gradient computation.

**Fix:** Auto-detect MXFP4 models and dequantize to bf16 using `Mxfp4Config(dequantize=True)`, following the [OpenAI fine-tuning cookbook](https://cookbook.openai.com/examples/partners/gpt-oss-20b-fine-tuning).

```python
_quant_cfg = getattr(_model_config, "quantization_config", None)
_is_mxfp4 = _quant_cfg and _quant_cfg.get("quant_method") == "mxfp4"
if _is_mxfp4:
    from transformers import Mxfp4Config
    model_init_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    model_init_kwargs["torch_dtype"] = torch.bfloat16
```

---

## Fix 2: Eager Attention for Training

**File:** `src/ctx_to_lora/model_loading.py`

**Problem:** GPT-OSS defaults to `kernels-community/vllm-flash-attn3` from the vLLM inference framework. This doesn't correctly handle packed sequences with per-sequence resetting `position_ids` during training, producing NaN outputs.

**Fix:** Force `eager` attention for GPT-OSS models.

```python
_is_gpt_oss = getattr(_model_config, "model_type", "") == "gpt_oss"
if _is_gpt_oss:
    model_init_kwargs["attn_implementation"] = "eager"
```

Also set `use_cache=False` globally (was `None`), required for training.

---

## Fix 3: Skip BitsAndBytes for MXFP4 Models

**File:** `src/ctx_to_lora/model_loading.py`

**Problem:** When `quantize_ctx_encoder=True`, the code applies BitsAndBytes NF4 quantization. But MXFP4 models already have a `quantization_config`, causing `ValueError: The model is quantized with Mxfp4Config but you are passing a BitsAndBytesConfig`.

**Fix:** Skip BitsAndBytes quantization when the model is MXFP4.

```python
if use_q_lora and not _is_mxfp4:
    bnb_config = BitsAndBytesConfig(...)
    model_init_kwargs["quantization_config"] = bnb_config
```

---

## Fix 4: LoRA Forward Pass for Quantized Layers

**File:** `src/ctx_to_lora/modeling/lora_layer.py`

**Problem:** `lora_forward` and `lora_forward_packed` called `torch.nn.Linear.forward(self, x)` directly. This bypasses dequantization logic in MXFP4 Linear subclasses that override `forward()`.

**Fix:** Use `self.base_layer(x)` which goes through the PEFT module's stored reference to the original layer, calling its potentially-overridden `forward()`.

```python
# Before:
base_out = torch.nn.Linear.forward(self, x, *args, **kwargs)

# After:
base_out = self.base_layer(x, *args, **kwargs)
```

---

## Fix 5: Einsum Dimension Mismatch (per_rank_gen)

**File:** `configs/main_exp/gpt_oss_20b.yaml`

**Problem:** `per_rank_gen` defaults to `False` in code, but all training scripts use `True`. When `False`, the aggregator squeezes the `r` dimension producing 4D output, but the `EinMix` head always expects 5D input: `"bs n_layers n_modules r d_latent -> ..."`. This causes einsum dimension mismatch at `hypernet.py:424`.

**Fix:** Explicitly set both flags in the config:

```yaml
per_rank_gen: true
per_layer_processing: true
```

`per_layer_processing: true` uses `ResMLPBlockPerLayer` which expects the 5D tensor from `per_rank_gen: true`.

---

## Fix 6: Save Override for Non-Tensor State Dict

**File:** `src/ctx_to_lora/trainer.py`

**Problem:** `ModulatedPretrainedModel.state_dict()` includes non-tensor items (`base_model_name_or_path` as string, `hypernet_config` and `ctx_encoder_args` as dataclasses). HF Trainer's default save uses safetensors which can only serialize tensors, causing `ValueError`.

**Fix:** Override `_save()` in `ModulatedModelTrainer` to use `torch.save()`.

```python
class ModulatedModelTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        if state_dict is None:
            state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
```

---

## Fix 7: Gradient Checkpointing Delegation

**File:** `src/ctx_to_lora/modeling/hypernet.py`

**Problem:** `ModulatedPretrainedModel` is a plain `nn.Module`, not `PreTrainedModel`. It doesn't have `gradient_checkpointing_enable()`/`disable()` methods. HF Trainer calls these when `gradient_checkpointing: true` is set, causing `AttributeError`.

**Fix:** Delegate to the inner `base_model`:

```python
def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
    self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

def gradient_checkpointing_disable(self):
    self.base_model.gradient_checkpointing_disable()
```

---

## Fix 8: Chat Template with `{% generation %}` Tags

**File:** `chat_templates/openai/gpt-oss-20b.jinja` (new)

**Problem:** GPT-OSS 20B's default chat template (from HuggingFace) lacks `{% generation %}` / `{% endgeneration %}` Jinja tags. The D2L data pipeline uses `return_assistant_tokens_mask=True` in `tokenizer.apply_chat_template()` to identify which tokens are assistant responses (and should have loss computed). Without `{% generation %}` tags, ALL `assistant_masks` are `False`, ALL labels become `-100` (IGNORE_INDEX), and cross-entropy loss is NaN (0/0 division).

**Symptom:** `ce_loss: nan`, `loss: 0`, `grad_norm: 0` -- but `gen_lora_l1_norm` is valid, meaning the hypernet generates LoRA weights but the base model forward produces no loss signal.

**Fix:** Created `chat_templates/openai/gpt-oss-20b.jinja` based on the HuggingFace template with `{% generation %}` tags added around assistant content:

```jinja
{%- elif loop.last and not add_generation_prompt %}
    {%- if "thinking" in message %}
        {{- "<|start|>assistant<|channel|>analysis<|message|>" }}{% generation %}{{ message.thinking + "<|end|>" }}{% endgeneration %}
    {%- endif %}
    {{- "<|start|>assistant<|channel|>final<|message|>" }}{% generation %}{{ message.content + "<|return|>" }}{% endgeneration %}
{%- else %}
    {{- "<|start|>assistant<|channel|>final<|message|>" }}{% generation %}{{ message.content + "<|end|>" }}{% endgeneration %}
{%- endif %}
```

**Important:** After changing the chat template, cached datasets must be deleted and re-tokenized:
```bash
rm -rf data/processed_datasets/
```

---

## Fix 9: LoRA Target Modules for MoE

**File:** `configs/main_exp/gpt_oss_20b.yaml`

**Problem:** GPT-OSS 20B's MLP uses Mixture-of-Experts with batched `nn.Parameter` tensors for expert weights. These are NOT `nn.Linear` modules, so LoRA injection (which patches `nn.Linear.forward`) cannot target them.

**Fix:** Target only attention modules:

```yaml
target_modules:
  - q_proj
```

Attention layers use standard `nn.Linear` and are fully compatible with D2L's LoRA injection.

---

## Training Config Summary

```yaml
# configs/main_exp/gpt_oss_20b.yaml
lora_r: 8
target_modules: [q_proj]          # attention only (MoE experts are nn.Parameter)
ctx_encoder_type: per_layer_activations
per_rank_gen: true                 # 5D aggregator output
per_layer_processing: true         # per-layer EinMix heads
use_kl_loss: false                 # no self-gen data needed
gradient_checkpointing: true
max_packed_inp_len: 1024
max_packed_ctx_len: 1024
```

---

## Hardware Requirements

| Component | Estimate |
|-----------|----------|
| GPT-OSS 20B base (bf16, dequantized) | ~42 GB |
| GPT-OSS 20B ctx encoder (bf16, before trim) | ~42 GB peak |
| After PerLayerActivations trim (~5 layers) | ~5 GB |
| Hypernet (fp32) + optimizer | ~2 GB |
| Activations (grad ckpt) | ~5-10 GB |
| **Peak during init** | **~90 GB** |
| **Steady state** | **~55 GB** |

Minimum: H200 SXM 141GB (single GPU, avoids multi-GPU batch size issues).

---

## Sanity Test Command

```bash
# Setup
bash scripts/setup_h100_pod.sh

# Clear cached datasets (required after chat template changes)
rm -rf data/processed_datasets/

# Run 1-step sanity test
WANDB_MODE=disabled .venv/bin/python -m accelerate.commands.launch train.py \
    configs/main_exp/gpt_oss_20b.yaml \
    --model_name_or_path=openai/gpt-oss-20b --max_steps=1

# Expected output:
# ce_loss: ~5.0 (valid, not NaN)
# grad_norm: ~130 (gradients flowing)
# gen_lora_l1_norm: ~0.001 (hypernet producing LoRA weights)
```

---

## Files Modified

| File | Change |
|------|--------|
| `src/ctx_to_lora/model_loading.py` | MXFP4 dequantization, eager attention, BnB skip |
| `src/ctx_to_lora/modeling/lora_layer.py` | `self.base_layer(x)` for quantized layers |
| `src/ctx_to_lora/modeling/hypernet.py` | gradient checkpointing delegation |
| `src/ctx_to_lora/trainer.py` | `_save()` override with `torch.save()` |
| `configs/main_exp/gpt_oss_20b.yaml` | New config for GPT-OSS 20B |
| `chat_templates/openai/gpt-oss-20b.jinja` | New chat template with `{% generation %}` tags |
| `scripts/setup_h100_pod.sh` | New setup script for H100/H200 pods |
