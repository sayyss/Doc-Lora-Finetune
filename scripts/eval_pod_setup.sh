#!/bin/bash
# Eval pod setup steps for GPT-OSS 20B

# 1. Clone repo (main branch has eval fixes)
git clone https://github.com/sayyss/Doc-Lora-Finetune.git && cd Doc-Lora-Finetune

# 2. Setup environment
bash scripts/setup_h100_pod.sh

# 3. Pin transformers (4.57 has overwrite_output_dir, _rotate_checkpoints compat)
uv pip install transformers==4.57.0

# 4. Download SQuAD (DROP and ROPES auto-download from HF Hub)
.venv/bin/python -c "from datasets import load_dataset; load_dataset('squad', cache_dir='data/raw_datasets/squad')"

# 5. Copy from training pod (replace <port> and <ip>):
#    scp -P <port> root@<ip>:~/Doc-to-LoRA/train_outputs/runs/<run_id>/checkpoint-<step>/pytorch_model.bin train_outputs/runs/<run_id>/checkpoint-<step>/
#    scp -P <port> root@<ip>:~/Doc-to-LoRA/train_outputs/runs/<run_id>/args.yaml train_outputs/runs/<run_id>/
#    scp -P <port> root@<ip>:~/Doc-to-LoRA/train_outputs/runs/<run_id>/cli_args.yaml train_outputs/runs/<run_id>/

# 6. Run eval
# .venv/bin/python run_eval.py --checkpoint_path train_outputs/runs/<run_id>/checkpoint-<step>/pytorch_model.bin --eval_batch_size_gen 4
