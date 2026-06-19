# ==============================
# HumanEval Multi-Stage Orchestration (LangGraph version)
# Planner: microsoft/phi-1_5
# Coder: deepseek-ai/deepseek-coder-1.3b-instruct
# Evaluator: pass@1
#
# Same 3-stage logic as MultiAgentCodeGeneration.py, expressed as a
# LangGraph StateGraph: planner_node -> coder_node -> evaluator_node.
# Models, prompts, and evaluation logic are unchanged.
# ==============================

# --- INSTALLATION (Colab) ---
# !pip install transformers datasets accelerate torch langgraph --quiet

import re
import json
import time
import tempfile
import subprocess
import os
from typing import TypedDict

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from langgraph.graph import StateGraph, END

# ==============================
# CONFIGURATION
# ==============================
MAX_TASKS = None  # None = full HumanEval
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# LOAD MODELS
# ==============================
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto" if DEVICE == "cuda" else None)
    model.eval()
    return tokenizer, model

print("Loading Planner (microsoft/phi-1_5)...")
planner_tokenizer, planner_model = load_model_and_tokenizer("microsoft/phi-1_5")

print("Loading Coder (deepseek-ai/deepseek-coder-1.3b-instruct)...")
coder_tokenizer, coder_model = load_model_and_tokenizer("deepseek-ai/deepseek-coder-1.3b-instruct")

# ==============================
# UTILITY FUNCTIONS (unchanged)
# ==============================
def generate_text(model, tokenizer, prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_code(text):
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def run_candidate(code, test_code):
    """Run candidate code + test code in isolated subprocess."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code + "\n" + test_code)
        tmp_path = f.name
    try:
        res = subprocess.run(
            ["python3", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res.returncode == 0:
            return "PASS"
        else:
            return f"FAIL: {res.stderr.strip().splitlines()[-1]}"
    except Exception as e:
        return f"ERROR: {str(e)}"
    finally:
        os.remove(tmp_path)


# ==============================
# LANGGRAPH STATE + NODES
# ==============================
class TaskState(TypedDict):
    task_id: str
    prompt: str
    test_code: str
    planner_output: str
    coder_output: str
    candidate_code: str
    verdict: str


def planner_node(state: TaskState) -> TaskState:
    planner_prompt = f"Provide a clear plan on how to implement the following function, step by step:\n\n{state['prompt']}"
    planner_output = generate_text(planner_model, planner_tokenizer, planner_prompt, max_new_tokens=256)
    print("\n📘 Planner Output:\n", planner_output.strip())
    return {**state, "planner_output": planner_output}


def coder_node(state: TaskState) -> TaskState:
    coder_prompt = (
        f"Using the following plan, write a correct Python implementation.\n"
        f"PLAN:\n{state['planner_output']}\n"
        f"PROMPT:\n{state['prompt']}\n"
        f"Return only code inside a single ```python``` block."
    )
    coder_output = generate_text(coder_model, coder_tokenizer, coder_prompt, max_new_tokens=512)
    candidate_code = extract_code(coder_output)
    print("\n💻 Coder Output:\n", candidate_code[:500])
    return {**state, "coder_output": coder_output, "candidate_code": candidate_code}


def evaluator_node(state: TaskState) -> TaskState:
    verdict = run_candidate(state["candidate_code"], state["test_code"])
    print("\n🧮 Verdict:", verdict)
    return {**state, "verdict": verdict}


graph = StateGraph(TaskState)
graph.add_node("planner", planner_node)
graph.add_node("coder", coder_node)
graph.add_node("evaluator", evaluator_node)
graph.set_entry_point("planner")
graph.add_edge("planner", "coder")
graph.add_edge("coder", "evaluator")
graph.add_edge("evaluator", END)

app = graph.compile()

# ==============================
# LOAD HUMAN EVAL DATASET + RUN
# ==============================
if __name__ == "__main__":
    dataset = load_dataset("openai_humaneval", split="test")
    results = []
    passed = 0
    start_time = time.time()

    for idx, task in enumerate(dataset):
        if MAX_TASKS and idx >= MAX_TASKS:
            break

        print("=" * 80)
        print(f"Task {idx + 1}: {task['task_id']}")
        print("-" * 80)
        print(task["prompt"])
        print("-" * 80)

        final_state = app.invoke(
            {
                "task_id": task["task_id"],
                "prompt": task["prompt"],
                "test_code": task["test"],
                "planner_output": "",
                "coder_output": "",
                "candidate_code": "",
                "verdict": "",
            }
        )

        if final_state["verdict"].startswith("PASS"):
            passed += 1

        results.append(final_state)
        print(f"📊 Running pass@1: {passed}/{len(results)} = {passed/len(results)*100:.2f}%\n")

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"Evaluation Complete: {len(results)} tasks")
    print(f"Passed: {passed} | Pass@1: {passed/len(results)*100:.2f}% | Time: {elapsed/60:.1f} min")

    with open("humaneval_orchestration_results_langgraph.json", "w") as f:
        json.dump(results, f, indent=2)
