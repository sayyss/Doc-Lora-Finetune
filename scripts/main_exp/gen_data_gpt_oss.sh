#!/bin/bash
set -e

# Generate self-gen data for standard QA datasets using GPT-OSS 20B
# Run on H200 SXM 141GB

.venv/bin/python data/self_generate_qa.py \
    --vllm_model openai/gpt-oss-20b \
    --ds_names squad_compact ropes_compact drop_compact \
    --split train \
    --closed_qa_prob 1.0 \
    --max_new_tokens 512

.venv/bin/python data/self_generate_qa.py \
    --vllm_model openai/gpt-oss-20b \
    --ds_names pwc_compact \
    --split train \
    --closed_qa_prob 0.0 \
    --max_new_tokens 512
