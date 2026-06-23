# ==============================================================
# orchestrator.py
#
# Single entry point that connects the Router Agent to the
# Planner -> Coder -> Debugger multi-agent pipeline.
#
# Flow:
#   1. User submits a free-text query (e.g. "write a function that ...")
#   2. Router classifies it: MPNet embedding similarity -> top-3
#      candidate categories -> gpt-4o-mini re-ranks to 1 final category
#   3. If the final category is one of the coding categories, the query
#      (treated as a HumanEval-style prompt) is routed into the
#      Planner -> Coder -> Debugger pipeline.
#   4. If it's any other category (e.g. project-planning categories),
#      the orchestrator declines, since only the code-generation
#      workflow is implemented end-to-end. This makes explicit which
#      branches are real vs. future work.
# ==============================================================

import re
import os
import ast
import json
import tempfile
import subprocess

import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------------------------------------------
# Router: same categories / embedding model as router.py
# ----------------------------------------------------------------
ROUTER_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CLASSIFIER_MODEL_NAME = "gpt-4o-mini"

CATEGORIES = [
    "Code Generation and Implementation",
    "Debugging and Error Handling",
    "Code Review and Optimization",
    "Testing and Quality Assurance",
    "Documentation and Explanation",
    "Research and Learning",
    "System Design and Architecture",
    "Data Analysis and Processing",
    "Project Scope Planning",
    "Resource Planning",
    "Schedule Planning",
    "Risk Planning",
    "Budget Planning",
    "Communication Planning",
    "Quality Planning",
    "Change Management Planning",
]

CATEGORY_DESCRIPTIONS = [
    "writing new code creating functions implementing algorithms building applications",
    "fixing bugs resolving errors troubleshooting exceptions stack traces",
    "code refactoring performance optimization improving code quality",
    "unit testing integration testing QA test automation",
    "documentation tutorials explaining programming concepts",
    "learning technologies researching programming topics",
    "system architecture microservices distributed systems scalability",
    "data analysis machine learning statistics visualization",
    "defining project scope objectives deliverables",
    "allocating people tools budget resources",
    "creating timelines milestones gantt charts",
    "identifying and mitigating project risks",
    "estimating project cost budgets",
    "stakeholder communication planning",
    "quality standards assurance processes",
    "handling project changes adapting scope schedule",
]

# Only these categories have an implemented downstream pipeline today.
# Everything else is classified correctly but explicitly declined,
# rather than silently mishandled.
CODE_PIPELINE_CATEGORIES = {
    "Code Generation and Implementation",
    "Debugging and Error Handling",
    "Code Review and Optimization",
    "Testing and Quality Assurance",
}


class Router:
    def __init__(self):
        self.embed_model = SentenceTransformer(ROUTER_MODEL_NAME)
        self.client = OpenAI()
        category_text = [f"{c} {d}" for c, d in zip(CATEGORIES, CATEGORY_DESCRIPTIONS)]
        self.category_embeddings = self.embed_model.encode(category_text, convert_to_tensor=True)

    def semantic_shortlist(self, query, k=3):
        query_embedding = self.embed_model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.category_embeddings)[0]
        top_k = torch.topk(scores, k=k)
        return [CATEGORIES[i] for i in top_k.indices]

    def llm_rerank(self, query, candidates):
        prompt = f"""
Classify the question into one of the following categories:

{candidates}

Question: {query}

Respond ONLY with the category name.
"""
        response = self.client.chat.completions.create(
            model=CLASSIFIER_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a classification assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def classify(self, query):
        candidates = self.semantic_shortlist(query)
        final_category = self.llm_rerank(query, candidates)
        return final_category, candidates


# ----------------------------------------------------------------
# Multi-agent coding pipeline: Planner -> Coder -> Debugger
# (same models / logic as MultiAgentCodeGeneration.py)
# ----------------------------------------------------------------
PLANNER_MODEL_NAME = "microsoft/phi-1_5"
CODER_MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
DEBUGGER_MODEL_NAME = "google/codegemma-2b"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYNTAX_ERROR_MARKERS = [
    "syntaxerror", "invalid syntax", "indentationerror", "taberror",
    "unexpected eof", "eof while parsing", "unterminated", "invalid token",
]


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto" if DEVICE == "cuda" else None
    )
    model.eval()
    return tokenizer, model


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
    return match.group(1).strip() if match else text.strip()


def run_candidate(code, test_code):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code + "\n" + test_code)
        tmp_path = f.name
    try:
        res = subprocess.run(
            ["python3", tmp_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if res.returncode == 0:
            return "PASS"
        err = res.stderr.strip().splitlines()
        return "FAIL: " + (err[-1] if err else res.stderr.strip())
    finally:
        os.remove(tmp_path)


def ast_equivalent(src1, src2):
    try:
        a1, a2 = ast.parse(src1), ast.parse(src2)
    except Exception:
        return False

    def _normalize(node):
        for field, value in list(ast.iter_fields(node)):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        _normalize(item)
            elif isinstance(value, ast.AST):
                _normalize(value)
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(node, attr):
                try:
                    delattr(node, attr)
                except Exception:
                    pass

    _normalize(a1)
    _normalize(a2)
    return ast.dump(a1, include_attributes=False) == ast.dump(a2, include_attributes=False)


class CodingPipeline:
    """Planner -> Coder -> (conditional) Debugger, same as MultiAgentCodeGeneration.py"""

    def __init__(self):
        print("Loading Planner (microsoft/phi-1_5)...")
        self.planner_tok, self.planner_model = load_model_and_tokenizer(PLANNER_MODEL_NAME)
        print("Loading Coder (deepseek-ai/deepseek-coder-1.3b-instruct)...")
        self.coder_tok, self.coder_model = load_model_and_tokenizer(CODER_MODEL_NAME)
        print("Loading Debugger (google/codegemma-2b)...")
        self.debugger_tok, self.debugger_model = load_model_and_tokenizer(DEBUGGER_MODEL_NAME)

    def plan(self, prompt):
        planner_prompt = f"Provide a clear plan on how to implement the following function, step by step:\n\n{prompt}"
        return generate_text(self.planner_model, self.planner_tok, planner_prompt, max_new_tokens=256)

    def code(self, prompt, planner_output):
        coder_prompt = (
            f"Using the following plan, write a correct Python implementation.\n"
            f"PLAN:\n{planner_output}\nPROMPT:\n{prompt}\n"
            "Return only code inside a single ```python``` block."
        )
        coder_output = generate_text(self.coder_model, self.coder_tok, coder_prompt, max_new_tokens=512)
        return extract_code(coder_output)

    def debug(self, buggy_code):
        prompt = f"""Fix ONLY syntax errors in the following Python code.
Do NOT change logic or function signatures. Return ONLY corrected code inside one ```python``` block.

```python
{buggy_code}
```"""
        fixed_raw = generate_text(self.debugger_model, self.debugger_tok, prompt, max_new_tokens=512)
        fixed_code = extract_code(fixed_raw)

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
                tf.write(fixed_code)
                tmp = tf.name
            import py_compile
            py_compile.compile(tmp, doraise=True)
        except Exception:
            return None
        finally:
            os.remove(tmp)

        return fixed_code if ast_equivalent(buggy_code, fixed_code) else None

    def generate_tests(self, prompt, candidate_code):
        """
        Generate test cases for a live query that has no ground-truth test
        (unlike HumanEval, which ships task["test"] for every problem).

        CAVEAT - self-grading risk: these tests are written by the same
        family of model that may have misunderstood the problem when writing
        the candidate code. If the model's understanding of the problem is
        wrong, the test it writes for itself can encode the same wrong
        understanding, and the candidate will "pass" a test that doesn't
        actually check correctness. This is NOT equivalent to HumanEval's
        independent, human-authored test oracle - treat verdicts from this
        path as weaker evidence than benchmark Pass@1 results.

        To reduce (not eliminate) this risk, the test-writing call is given
        ONLY the original prompt - not the candidate code - so it can't
        simply encode whatever the candidate happens to do.
        """
        test_gen_prompt = f"""Write Python unit tests (using plain `assert` statements,
no pytest/unittest framework) for the following function specification.
Base the tests only on the specification below - do not assume any particular
implementation. Return ONLY code, no explanation.

SPECIFICATION:
{prompt}
"""
        response = self.router.client.chat.completions.create(
            model=CLASSIFIER_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a careful test-writing assistant."},
                {"role": "user", "content": test_gen_prompt},
            ],
            temperature=0,
        )
        return extract_code(response.choices[0].message.content.strip())

    def run(self, prompt, test_code=None):
        planner_output = self.plan(prompt)
        candidate_code = self.code(prompt, planner_output)

        test_source = "ground_truth"
        if test_code is None:
            # No ground-truth test available (live query, not a benchmark
            # problem) - fall back to LLM-generated tests. See
            # generate_tests() docstring for the self-grading caveat.
            test_code = self.generate_tests(prompt, candidate_code)
            test_source = "llm_generated"

        verdict = run_candidate(candidate_code, test_code)
        if verdict.lower().startswith("fail") and any(m in verdict.lower() for m in SYNTAX_ERROR_MARKERS):
            fixed = self.debug(candidate_code)
            if fixed is not None:
                candidate_code = fixed
                verdict = run_candidate(candidate_code, test_code)

        return {
            "planner_output": planner_output,
            "candidate_code": candidate_code,
            "test_code": test_code,
            "test_source": test_source,
            "verdict": verdict,
        }


# ----------------------------------------------------------------
# Orchestrator: Router -> dispatch -> CodingPipeline
# ----------------------------------------------------------------
class Orchestrator:
    def __init__(self):
        self.router = Router()
        self._coding_pipeline = None  # lazy-loaded, since it loads 3 LLMs

    @property
    def coding_pipeline(self):
        if self._coding_pipeline is None:
            self._coding_pipeline = CodingPipeline()
        return self._coding_pipeline

    def direct_answer(self, query, category):
        """Fallback for categories without a dedicated multi-agent pipeline:
        answer directly with a single LLM call, using the category as light
        context for the system prompt."""
        response = self.router.client.chat.completions.create(
            model=CLASSIFIER_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. The user's question has been "
                    f"classified as relating to: {category}. Answer it directly and concisely.",
                },
                {"role": "user", "content": query},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def handle(self, query, test_code=None):
        category, candidates = self.router.classify(query)
        print(f"Router candidates: {candidates}")
        print(f"Final category: {category}")

        if category not in CODE_PIPELINE_CATEGORIES:
            answer = self.direct_answer(query, category)
            return {"category": category, "routed_to": "direct_llm", "result": answer}

        result = self.coding_pipeline.run(query, test_code=test_code)
        return {"category": category, "routed_to": "coding_pipeline", "result": result}


if __name__ == "__main__":
    orchestrator = Orchestrator()
    while True:
        q = input("\nEnter query (or 'exit'): ")
        if q.lower() == "exit":
            break
        output = orchestrator.handle(q)
        print(json.dumps(output, indent=2, default=str))
