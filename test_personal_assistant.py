"""
Test Doc-to-LoRA for Personal Assistant Context Learning

Evaluates D2L's context internalization as a building block for a local
personal assistant that "grows with the user" — remembering personal info,
preferences, and tasks without stuffing everything into the prompt each time.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from ctx_to_lora.data.processing import tokenize_ctx_text
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling import hypernet

sys.modules["ctx_to_lora.modeling_utils"] = hypernet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str):
    """Load the hypernetwork checkpoint and return model + tokenizers."""
    from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=False)
    modulated_model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_flash_attn=True, use_sequence_packing=False
    )
    modulated_model = modulated_model.to(device).to(torch.bfloat16)
    modulated_model.eval()

    ctx_encoder_model_name_or_path = (
        modulated_model.ctx_encoder_args.ctx_encoder_model_name_or_path
        or modulated_model.base_model.config.name_or_path
    )
    ctx_tokenizer = get_tokenizer(ctx_encoder_model_name_or_path, train=False)
    base_tokenizer = get_tokenizer(
        modulated_model.base_model.config.name_or_path, train=False
    )

    # Load custom chat template for Gemma
    model_name = modulated_model.base_model.config.name_or_path
    if "gemma" in model_name.lower():
        template_path = "chat_templates/google/gemma-2-2b-it.jinja"
        if os.path.exists(template_path):
            with open(template_path) as f:
                base_tokenizer.chat_template = f.read()
                print(f"Loaded custom chat template from {template_path}")

    print(f"Model loaded: {model_name}")
    return modulated_model, ctx_tokenizer, base_tokenizer


def process_context(context: str, ctx_tokenizer):
    """Tokenize and prepare context for the hypernetwork."""
    context = context.strip() if context else ""
    tokenized_contexts = tokenize_ctx_text({"context": [context]}, ctx_tokenizer)
    ctx_ids = tokenized_contexts["ctx_ids"]
    ctx_ids = [
        torch.tensor(ctx_id, dtype=torch.long, device=device) for ctx_id in ctx_ids
    ]
    ctx_attn_mask = [torch.ones_like(ids) for ids in ctx_ids]
    ctx_ids = torch.nn.utils.rnn.pad_sequence(
        ctx_ids, batch_first=True, padding_value=0
    )
    ctx_attn_mask = torch.nn.utils.rnn.pad_sequence(
        ctx_attn_mask, batch_first=True, padding_value=0
    )
    return {"ctx_ids": ctx_ids, "ctx_attn_mask": ctx_attn_mask}


def ask(
    model, ctx_tokenizer, base_tokenizer,
    context: str, chat_history: list[dict],
    question: str,
    context_scaler: float = 1.0,
    bias_scaler: float = 1.0,
) -> str:
    """Send a question and return the model's response."""
    chat_history.append({"role": "user", "content": question})

    with torch.inference_mode(), torch.amp.autocast(str(device)):
        ctx_inputs = process_context(context, ctx_tokenizer)
        ctx_ids = ctx_inputs["ctx_ids"].to(device)
        ctx_attn_mask = ctx_inputs["ctx_attn_mask"].to(device)

        scalers_tensor = torch.tensor(
            [context_scaler], dtype=torch.float32, device=device
        )

        model_inputs = base_tokenizer.apply_chat_template(
            chat_history, return_tensors="pt", add_generation_prompt=True
        ).to(device)

        outputs = model.generate(
            ctx_ids=ctx_ids,
            ctx_attn_mask=ctx_attn_mask,
            n_ctx_chunks=torch.tensor([len(ctx_ids)], device=ctx_ids.device),
            scalers=scalers_tensor,
            bias_scaler=bias_scaler,
            input_ids=model_inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0,
        )

        response = base_tokenizer.decode(
            outputs[0][model_inputs.shape[1]:], skip_special_tokens=True
        )

    chat_history.append({"role": "assistant", "content": response})
    return response


# ── Test Scenarios ───────────────────────────────────────────────────────────

def test_a_basic_fact_recall(model, ctx_tokenizer, base_tokenizer, results):
    """Test A: Basic Fact Recall (Identity & Preferences)"""
    print("\n" + "=" * 70)
    print("TEST A: Basic Fact Recall (Identity & Preferences)")
    print("=" * 70)

    context = (
        "My name is Alex Chen. I'm 28 years old, a software engineer at Meridian Labs. "
        "I live in Portland, Oregon. My partner's name is Sam. I have a golden retriever "
        "named Biscuit. I prefer dark mode in all apps. My favorite programming language "
        "is Rust. I take my coffee black, no sugar. My birthday is March 15th."
    )

    questions = [
        ("What's my name?", "Alex Chen"),
        ("What's my dog's name?", "Biscuit"),
        ("How do I take my coffee?", "black, no sugar"),
        ("When is my birthday?", "March 15"),
        ("Where do I work?", "Meridian Labs"),
    ]

    test_results = []
    chat_history = [{"role": "system", "content": ""}]

    for question, expected_keyword in questions:
        response = ask(model, ctx_tokenizer, base_tokenizer, context, chat_history, question)
        correct = expected_keyword.lower() in response.lower()
        test_results.append({
            "question": question,
            "expected": expected_keyword,
            "response": response,
            "correct": correct,
        })
        status = "PASS" if correct else "FAIL"
        print(f"  [{status}] Q: {question}")
        print(f"         A: {response.strip()[:120]}")
        # Reset chat history for each question (independent recall)
        chat_history = [{"role": "system", "content": ""}]

    accuracy = sum(1 for r in test_results if r["correct"]) / len(test_results)
    print(f"\n  Accuracy: {accuracy:.0%} ({sum(1 for r in test_results if r['correct'])}/{len(test_results)})")

    results["test_a"] = {
        "name": "Basic Fact Recall",
        "context_length": len(context),
        "results": test_results,
        "accuracy": accuracy,
    }
    return accuracy


def test_b_dense_fact_retention(model, ctx_tokenizer, base_tokenizer, results):
    """Test B: Dense Fact Retention (Stress Test)"""
    print("\n" + "=" * 70)
    print("TEST B: Dense Fact Retention (Stress Test — 20+ facts)")
    print("=" * 70)

    context = (
        "My name is Alex Chen. I'm 28 years old. I'm a software engineer at Meridian Labs. "
        "I live at 742 Evergreen Terrace, Portland, Oregon 97201. "
        "My phone number is 503-555-0147. My email is alex.chen@meridian.io. "
        "My partner's name is Sam Rivera. Sam is a graphic designer. "
        "I have a golden retriever named Biscuit, age 4. "
        "My best friend is Jordan Park. Jordan's birthday is July 22nd. "
        "My manager at work is Diana Frost. "
        "My dentist is Dr. Patel at SmileCare Dental. "
        "My favorite restaurant is Bamboo Garden on 5th Avenue. "
        "I drive a 2019 blue Subaru Outback. License plate: XYZ-4829. "
        "My gym is Portland Fitness on Oak Street. I go to yoga on Tuesdays. "
        "My Wi-Fi password is Biscuit2020. "
        "I'm allergic to shellfish. "
        "My blood type is O-positive. "
        "My emergency contact is Sam Rivera at 503-555-0198. "
        "My favorite programming language is Rust. My second favorite is Python. "
        "I prefer dark mode. I use NeoVim as my editor. "
        "I'm learning Japanese — currently at JLPT N4 level. "
        "My birthday is March 15, 1997."
    )

    questions = [
        ("What is my address?", "742 Evergreen"),
        ("What's my phone number?", "503-555-0147"),
        ("What does my partner Sam do for work?", "graphic designer"),
        ("How old is my dog?", "4"),
        ("Who is my best friend?", "Jordan"),
        ("Who is my manager?", "Diana Frost"),
        ("What car do I drive?", "Subaru"),
        ("What am I allergic to?", "shellfish"),
        ("What's my blood type?", "O-positive"),
        ("What editor do I use?", "NeoVim"),
        ("What language am I learning?", "Japanese"),
        ("What's my favorite restaurant?", "Bamboo Garden"),
        ("What's my license plate number?", "XYZ-4829"),
        ("Who is my emergency contact?", "Sam"),
        ("What is my email?", "alex.chen@meridian"),
    ]

    test_results = []
    for question, expected_keyword in questions:
        chat_history = [{"role": "system", "content": ""}]
        response = ask(model, ctx_tokenizer, base_tokenizer, context, chat_history, question)
        correct = expected_keyword.lower() in response.lower()
        test_results.append({
            "question": question,
            "expected": expected_keyword,
            "response": response,
            "correct": correct,
        })
        status = "PASS" if correct else "FAIL"
        print(f"  [{status}] Q: {question}")
        print(f"         A: {response.strip()[:120]}")

    accuracy = sum(1 for r in test_results if r["correct"]) / len(test_results)
    print(f"\n  Coverage: {accuracy:.0%} ({sum(1 for r in test_results if r['correct'])}/{len(test_results)})")

    results["test_b"] = {
        "name": "Dense Fact Retention (20+ facts)",
        "context_length": len(context),
        "num_facts_tested": len(questions),
        "results": test_results,
        "accuracy": accuracy,
    }
    return accuracy


def test_c_reasoning(model, ctx_tokenizer, base_tokenizer, results):
    """Test C: Reasoning Over Internalized Info"""
    print("\n" + "=" * 70)
    print("TEST C: Reasoning Over Internalized Info (Schedule)")
    print("=" * 70)

    context = (
        "My schedule for today:\n"
        "9:00 AM - Standup with backend team (Zoom)\n"
        "10:30 AM - Code review for PR #482\n"
        "12:00 PM - Lunch with Sam at Bamboo Garden\n"
        "2:00 PM - 1:1 with manager (Room 3B)\n"
        "4:00 PM - Dentist appointment at SmileCare Dental\n"
        "6:30 PM - Yoga class at Portland Fitness"
    )

    questions = [
        ("What am I doing at noon?", ["lunch", "sam", "bamboo"]),
        ("Do I have any evening plans?", ["yoga", "6:30", "portland fitness"]),
        ("When is my next meeting after standup?", ["10:30", "code review", "PR"]),
        ("Am I free at 3pm?", ["no", "not", "dentist", "between", "2:00", "4:00"]),
    ]

    test_results = []
    for question, expected_keywords in questions:
        chat_history = [{"role": "system", "content": ""}]
        response = ask(model, ctx_tokenizer, base_tokenizer, context, chat_history, question)
        response_lower = response.lower()
        correct = any(kw.lower() in response_lower for kw in expected_keywords)
        test_results.append({
            "question": question,
            "expected_keywords": expected_keywords,
            "response": response,
            "correct": correct,
        })
        status = "PASS" if correct else "FAIL"
        print(f"  [{status}] Q: {question}")
        print(f"         A: {response.strip()[:150]}")

    accuracy = sum(1 for r in test_results if r["correct"]) / len(test_results)
    print(f"\n  Reasoning accuracy: {accuracy:.0%} ({sum(1 for r in test_results if r['correct'])}/{len(test_results)})")

    results["test_c"] = {
        "name": "Reasoning Over Internalized Info",
        "context_length": len(context),
        "results": test_results,
        "accuracy": accuracy,
    }
    return accuracy


def test_d_multi_turn(model, ctx_tokenizer, base_tokenizer, results):
    """Test D: Multi-turn Conversation Coherence"""
    print("\n" + "=" * 70)
    print("TEST D: Multi-turn Conversation Coherence")
    print("=" * 70)

    context = (
        "My name is Alex Chen. I'm a software engineer at Meridian Labs. "
        "My partner's name is Sam. I have a golden retriever named Biscuit. "
        "I live in Portland, Oregon. My favorite programming language is Rust."
    )

    # Use a SINGLE chat history across all turns
    chat_history = [{"role": "system", "content": ""}]

    turns = [
        ("What's my name?", "Alex"),
        ("And where do I work?", "Meridian"),
        ("What's my dog's name?", "Biscuit"),
        ("What programming language do I prefer?", "Rust"),
        ("Remind me, where do I live?", "Portland"),
    ]

    test_results = []
    for question, expected_keyword in turns:
        response = ask(model, ctx_tokenizer, base_tokenizer, context, chat_history, question)
        correct = expected_keyword.lower() in response.lower()
        test_results.append({
            "question": question,
            "expected": expected_keyword,
            "response": response,
            "correct": correct,
        })
        status = "PASS" if correct else "FAIL"
        print(f"  [{status}] Turn {len(test_results)}: {question}")
        print(f"         A: {response.strip()[:120]}")

    accuracy = sum(1 for r in test_results if r["correct"]) / len(test_results)
    print(f"\n  Multi-turn coherence: {accuracy:.0%} ({sum(1 for r in test_results if r['correct'])}/{len(test_results)})")

    results["test_d"] = {
        "name": "Multi-turn Conversation Coherence",
        "num_turns": len(turns),
        "results": test_results,
        "accuracy": accuracy,
    }
    return accuracy


def test_e_context_scaling(model, ctx_tokenizer, base_tokenizer, results):
    """Test E: Context Scaling Sensitivity"""
    print("\n" + "=" * 70)
    print("TEST E: Context Scaling Sensitivity")
    print("=" * 70)

    context = (
        "My name is Alex Chen. I work at Meridian Labs. "
        "My dog is a golden retriever named Biscuit."
    )

    question = "What's my dog's name?"
    expected = "biscuit"

    scaler_configs = [
        {"context_scaler": 0.0, "bias_scaler": 1.0, "label": "ctx=0.0 (no context)"},
        {"context_scaler": 0.5, "bias_scaler": 1.0, "label": "ctx=0.5 (half)"},
        {"context_scaler": 1.0, "bias_scaler": 1.0, "label": "ctx=1.0 (default)"},
        {"context_scaler": 1.5, "bias_scaler": 1.0, "label": "ctx=1.5 (amplified)"},
        {"context_scaler": 1.0, "bias_scaler": 0.0, "label": "ctx=1.0, bias=0.0"},
        {"context_scaler": 1.0, "bias_scaler": 1.5, "label": "ctx=1.0, bias=1.5"},
    ]

    test_results = []
    for config in scaler_configs:
        chat_history = [{"role": "system", "content": ""}]
        response = ask(
            model, ctx_tokenizer, base_tokenizer, context, chat_history, question,
            context_scaler=config["context_scaler"],
            bias_scaler=config["bias_scaler"],
        )
        correct = expected in response.lower()
        test_results.append({
            "config": config["label"],
            "response": response,
            "correct": correct,
        })
        status = "PASS" if correct else "FAIL"
        print(f"  [{status}] {config['label']}")
        print(f"         A: {response.strip()[:120]}")

    results["test_e"] = {
        "name": "Context Scaling Sensitivity",
        "question": question,
        "results": test_results,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    checkpoint_path = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print("=" * 70)
    print("Doc-to-LoRA Personal Assistant Context Learning Test")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    model, ctx_tokenizer, base_tokenizer = load_model(checkpoint_path)

    results = {
        "meta": {
            "checkpoint": checkpoint_path,
            "device": str(device),
            "timestamp": datetime.now().isoformat(),
            "model_name": model.base_model.config.name_or_path,
        }
    }

    start_time = time.time()

    test_a_basic_fact_recall(model, ctx_tokenizer, base_tokenizer, results)
    test_b_dense_fact_retention(model, ctx_tokenizer, base_tokenizer, results)
    test_c_reasoning(model, ctx_tokenizer, base_tokenizer, results)
    test_d_multi_turn(model, ctx_tokenizer, base_tokenizer, results)
    test_e_context_scaling(model, ctx_tokenizer, base_tokenizer, results)

    elapsed = time.time() - start_time

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for key in ["test_a", "test_b", "test_c", "test_d", "test_e"]:
        r = results[key]
        if "accuracy" in r:
            print(f"  {r['name']}: {r['accuracy']:.0%}")
        else:
            passed = sum(1 for x in r["results"] if x["correct"])
            print(f"  {r['name']}: {passed}/{len(r['results'])} configs recalled correctly")

    print(f"\n  Total time: {elapsed:.1f}s")

    # Save results
    output_path = Path("test_results_personal_assistant.json")
    results["meta"]["elapsed_seconds"] = elapsed
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
