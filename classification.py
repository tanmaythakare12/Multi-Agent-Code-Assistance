from openai import OpenAI

client = OpenAI()

def prompt_injection(input_question: str) -> str:
    return f"""
Classify the following question into one of these categories:
["Code Generation and Implementation - Writing new code, creating functions, implementing algorithms, building applications",
"Debugging and Error Handling - Fixing bugs, resolving errors, troubleshooting issues, exception handling",
"Code Review and Optimization - Analyzing code quality, performance optimization, refactoring, code improvement",
"Testing and Quality Assurance - Writing tests, unit testing, integration testing, test automation, QA processes",
"Documentation and Explanation - Writing documentation, explaining concepts, creating tutorials, code comments",
"Research and Learning - Learning new technologies, understanding concepts, finding resources, educational content",
"System Design and Architecture - Designing systems, architectural decisions, scalability, infrastructure",
"Data Analysis and Processing - Data manipulation, analysis, visualization, machine learning, statistics",
"Project Scope Planning - Defining project objectives, deliverables, and boundaries",
    "Resource Planning - Allocating people, tools, budget, and time effectively",
    "Schedule Planning - Creating timelines, milestones, and Gantt charts",
    "Risk Planning - Identifying, assessing, and mitigating potential project risks",
    "Budget Planning - Estimating costs, preparing budgets, and controlling expenditures",
    "Communication Planning - Defining stakeholder communication strategies and channels",
    "Quality Planning - Setting quality standards and quality assurance processes",
    "Change Management Planning - Planning for adapting project scope, schedule, and resources"
]

Question: {input_question}

Only respond with the category name, nothing else.
"""
while True:
    question=input("Enter your programming-related question: ")
    prompt = prompt_injection(question)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a classification assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0  # Low randomness for deterministic classification
    )

    print(response.choices[0].message.content.strip())
    # exit loop if user types 'exit'
    if question.lower() == 'exit':
        break
