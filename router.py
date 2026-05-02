from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import torch

# initialize models
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
client = OpenAI()

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
"Change Management Planning"
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
"handling project changes adapting scope schedule"
]

category_text = [
f"{c} {d}" for c,d in zip(categories, category_descriptions)
]

category_embeddings = embed_model.encode(category_text, convert_to_tensor=True)

def semantic_router(query):

    query_embedding = embed_model.encode(query, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, category_embeddings)[0]

    top_k = torch.topk(scores, k=3)

    candidates = [categories[i] for i in top_k.indices]

    return candidates


def llm_classifier(query, candidate_categories):

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
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


while True:

    question = input("\nEnter question: ")

    if question.lower() == "exit":
        break

    candidates = semantic_router(question)

    print("Top semantic candidates:", candidates)

    final_category = llm_classifier(question, candidates)

    print("Final Category:", final_category)
