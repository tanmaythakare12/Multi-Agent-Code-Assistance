# Adaptive Multi-Agent Orchestrator

A role-aware LLM orchestration framework for AI-assisted software engineering. Instead of relying on a single monolithic model, AMO routes tasks through specialized agents that collaborate to improve code correctness, planning quality, and debug robustness — evaluated on HumanEval, HumanEval+, and PlanBench.

**Highlights:**
- Multi-agent pipeline improves Pass@1 by **+38.8%** over the DeepSeek 1.3B single-model baseline
- Two-stage semantic router covers 16 task categories, shortlisting top 3 via MPNet embeddings before LLM re-ranking
- Sandboxed code execution feeds runtime errors back to the LLM for targeted repairs, contributing a further **+6.7% Pass@1** improvement

---

## System Architecture

### End-to-End Execution Flow

```
User → Router → Planner → Generator → Tester → Debugger → Final Output
```

The Router classifies incoming questions into one of 16 task categories and selects the appropriate workflow. If the Tester detects failures, a debug-feedback loop fires before producing the final result.

**Supported workflows:**

- Code Generation
- Debugging & Error Handling
- Testing & QA
- Research & Learning
- Code Review / QA

Each workflow is modular and role-specific.

### Agent Roles

| Agent | Model | Responsibility |
|---|---|---|
| Planner | `microsoft/phi-1_5` | Step-by-step decomposition of the task |
| Generator (Coder) | `deepseek-ai/deepseek-coder-1.3b-instruct` | Produces the Python implementation from the plan |
| Debugger | `google/codegemma-2b` | Fixes syntax errors; accepts fix only if AST is unchanged |
| Router / Classifier | `gpt-4o-mini` | Re-ranks top-3 semantic candidates and selects final workflow |
| Semantic Embeddings | `sentence-transformers/all-mpnet-base-v2` (MPNet) | Shortlists top 3 of 16 categories via cosine similarity |
| Baselines | `TinyLlama-1.1B`, `microsoft/phi-2`, `microsoft/phi-1_5` | Single-model comparison |

---

## Repository Structure

```
Multi-Agent-Code-Assistance-master/
│
├── MultiAgentCodeGeneration.py          # Core 3-agent pipeline (Planner → Coder → Evaluator)
├── MultiAgentCodeGeneration.ipynb       # Notebook version of the above
│
├── CoderAgentWithDebugFeedback.ipynb    # Extended pipeline with Debugger agent + AST check
│
├── Baselines_fixed.ipynb                # Single-model baselines (TinyLlama, Phi-2) for comparison
├── microsoft_phi_baseline.ipynb         # Dedicated Phi-1.5 baseline evaluation
│
├── classification.py                    # GPT-4o-mini prompt-injection question classifier
├── router.py                            # Semantic router + LLM re-ranker for question routing
│
└── requirements.txt                     # Python dependencies
```

---

## Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (CPU fallback is available but slow)
- OpenAI API key (for `classification.py` and `router.py`)

### Install

```bash
pip install -r requirements.txt
```

For Google Colab, each notebook includes inline install cells:

```python
!pip install transformers datasets accelerate torch --quiet
```

### Environment

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Usage

### Run the multi-agent pipeline

```bash
python MultiAgentCodeGeneration.py
```

Iterates over the full HumanEval test set, printing per-task verdicts and a running `pass@1` score. Results are saved to `humaneval_orchestration_results.json`.

To limit tasks for quick testing, set `MAX_TASKS` at the top of the file:

```python
MAX_TASKS = 10  # None = full HumanEval (164 tasks)
```

### Run the question classifier

```bash
python classification.py
```

Interactive loop — enter a question, get a category. Type `exit` to quit.

### Run the semantic router

```bash
python router.py
```

Same interactive loop, but uses a two-stage approach: MPNet embeddings shortlist the top 3 of 16 categories via cosine similarity, then GPT-4o-mini re-ranks and selects the final category.

---

## Evaluation & Results

**Benchmarks:** HumanEval, HumanEval+, PlanBench  
**Metrics:** Pass@1 (code correctness), BERTScore (planning quality)

### Code Generation — Performance Progression (Pass@1)

| System | Pass@1 |
|---|---|
| Single-model baseline (DeepSeek 1.3B) | 42.30% |
| Planner + Generator | 74.39% |
| Full Multi-Agent + Debug Loop | 81.10% |

**+38.8 percentage points absolute** / **+91.7% relative** improvement over the DeepSeek 1.3B baseline.

The debug-feedback loop alone accounts for **+6.7% Pass@1** by feeding sandboxed runtime errors back to the LLM for targeted repairs rather than full regeneration. Structured planning is the other primary contributor, reducing logical and structural errors earlier in the pipeline.

---

## Question Classification Categories

Both `classification.py` and `router.py` classify questions into 16 categories before routing:

**Code-focused:** Code Generation & Implementation · Debugging & Error Handling · Code Review & Optimization · Testing & QA · Documentation & Explanation · Research & Learning · System Design & Architecture · Data Analysis & Processing

**Project-focused:** Project Scope Planning · Resource Planning · Schedule Planning · Risk Planning · Budget Planning · Communication Planning · Quality Planning · Change Management Planning

---

## Key Insights

1. No single LLM performs best across all reasoning domains.
2. Planning significantly reduces logical and structural errors in generated code.
3. Debug-feedback loops improve robustness and reduce hallucination.
4. Smaller models combined via orchestration outperform larger monolithic systems.

---

## Future Work

- Reinforcement-based routing optimization
- Self-learning debugger agent
- Adaptive pipeline construction
- Meta-routing confidence tuning
- Continual memory integration

---

## Notes

- All generation runs at `temperature=0` (greedy decoding) for deterministic, reproducible results.
- Generated code is executed in a temporary subprocess. **Do not run on untrusted inputs in production** — no additional sandboxing is applied beyond process isolation.
- The Debugger agent only accepts repairs where the fixed code produces an identical AST, preventing silent logic rewrites.

---

## Conclusion

AMO demonstrates that reliability in AI-assisted software engineering emerges from structured, modular, and verifiable multi-agent workflows rather than increasing model size alone. Across benchmarks, the orchestrated pipeline consistently outperforms monolithic LLM baselines in correctness, robustness, and planning quality.
