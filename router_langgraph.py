"""
LangGraph + LlamaIndex re-implementation of router.py

Same two-stage logic as the original:
  1. Semantic retrieval over category descriptions (MPNet embeddings)
  2. LLM re-classification among the top-k candidates (GPT-4o-mini)

Original logic is preserved; it's now expressed as:
  - a LlamaIndex VectorStoreIndex for the embedding/retrieval step
  - a LangGraph StateGraph (2 nodes) for orchestration
"""

from typing import TypedDict, List

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langgraph.graph import StateGraph, END
from openai import OpenAI

client = OpenAI()

# ==============================
# CATEGORY DEFINITIONS (unchanged from router.py)
# ==============================
categories = [
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

category_descriptions = [
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

# ==============================
# LLAMAINDEX RETRIEVAL LAYER
# (replaces the manual SentenceTransformer + cos_sim + topk logic)
# ==============================
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

documents = [
    Document(text=f"{c} {d}", metadata={"category": c})
    for c, d in zip(categories, category_descriptions)
]

index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)


def semantic_router(query: str) -> List[str]:
    """Top-3 candidate categories via LlamaIndex vector retrieval."""
    nodes = retriever.retrieve(query)
    return [n.node.metadata["category"] for n in nodes]


def llm_classifier(query: str, candidate_categories: List[str]) -> str:
    """Unchanged GPT-4o-mini re-ranking step."""
    prompt = f"""
Classify the question into one of the following categories:

{candidate_categories}

Question: {query}

Respond ONLY with the category name.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a classification assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ==============================
# LANGGRAPH ORCHESTRATION
# ==============================
class RouterState(TypedDict):
    query: str
    candidates: List[str]
    final_category: str


def route_node(state: RouterState) -> RouterState:
    candidates = semantic_router(state["query"])
    print("Top semantic candidates:", candidates)
    return {**state, "candidates": candidates}


def classify_node(state: RouterState) -> RouterState:
    final_category = llm_classifier(state["query"], state["candidates"])
    print("Final Category:", final_category)
    return {**state, "final_category": final_category}


graph = StateGraph(RouterState)
graph.add_node("route", route_node)
graph.add_node("classify", classify_node)
graph.set_entry_point("route")
graph.add_edge("route", "classify")
graph.add_edge("classify", END)

app = graph.compile()


if __name__ == "__main__":
    while True:
        question = input("\nEnter question: ")
        if question.lower() == "exit":
            break
        app.invoke({"query": question, "candidates": [], "final_category": ""})
