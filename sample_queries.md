# Sample Queries & Evaluation Benchmark

This document records the evaluation of the RAG Chatbot on the **"Agentic AI" eBook** (`Ebook-Agentic-AI.pdf`) across diverse queries. Each sample showcases the exact request, response payload, confidence score, and retrieved source chunks with page citations.

---

## Summary of Test Cases

| # | Test Question | Type | Confidence | Decision |
|---|---|---|---|---|
| 1 | `What is Agentic AI?` | Core Concept Definition | **0.8168** (High) | `generate` |
| 2 | `How is Agentic AI different from traditional AI or generative AI?` | Comparative Analysis | **0.6663** (High) | `generate` |
| 3 | `What are the key components of an AI agent?` | Architecture / Pillars | **0.7473** (High) | `generate` |
| 4 | `What are the practical applications of Agentic AI in enterprises?` | Enterprise Use Cases | **0.7798** (High) | `generate` |
| 5 | `How do Multi-Agent Systems work?` | Systems & Collaboration | **0.7832** (High) | `generate` |
| 6 | `Who won the FIFA World Cup in 2018?` | **Out-of-Scope (Grounding Test)** | **0.1096** (Low) | `fallback` |

---

## Query 1: Core Definition

### Request
```http
POST /chat HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "question": "What is Agentic AI?"
}
```

### Response
```json
{
  "answer": "Agentic AI refers to systems capable of autonomous decision-making, goal pursuit, and taking independent actions to achieve specified outcomes. Unlike reactive tools, it acts proactively by planning, executing, and adapting based on environment feedback.",
  "retrieved_chunks": [
    {
      "page": 3,
      "score": 0.8168,
      "text": "Agentic AI\nAn Executive's Guide to In-depth\nUnderstanding of Agentic AI"
    },
    {
      "page": 7,
      "score": 0.7812,
      "text": "In this section, we will define what Agentic AI is and, more importantly, what it's not, as it's often misunderstood."
    },
    {
      "page": 32,
      "score": 0.7749,
      "text": "Together, they help these systems make decisions, learn from experience, and interact with their environment effectively."
    },
    {
      "page": 18,
      "score": 0.7557,
      "text": "A Journey into the Heart of Autonomous Intelligence\nAgentic AI refers to systems capable of autonomous decision-making..."
    }
  ],
  "confidence": 0.8168
}
```

**Observation:** Top chunk score (0.8168) is well above the similarity threshold (0.30), correctly routing to `generate`.

---

## Query 2: Comparative Analysis

### Request
```http
POST /chat HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "question": "How is Agentic AI different from traditional AI or generative AI?"
}
```

### Response
```json
{
  "answer": "Traditional AI and generative AI primarily respond to direct prompts or perform predefined tasks reactively. In contrast, Agentic AI introduces autonomy, planning, and goal pursuit—shifting technology from reactive execution to proactive decision-making.",
  "retrieved_chunks": [
    {
      "page": 3,
      "score": 0.6663,
      "text": "Agentic AI\nAn Executive's Guide to In-depth\nUnderstanding of Agentic AI"
    },
    {
      "page": 7,
      "score": 0.6630,
      "text": "In this section, we will define what Agentic AI is and, more importantly, what it's not, as it's often misunderstood."
    },
    {
      "page": 8,
      "score": 0.6275,
      "text": "Understanding the Shift from Reactive to Proactive Technology\nImagine Sarah, a busy entrepreneur juggling multiple projects..."
    },
    {
      "page": 10,
      "score": 0.6124,
      "text": "The table below compares the key aspects of LLMs and agents: How They Work Together? Agentic AI and LLMs collaborate..."
    }
  ],
  "confidence": 0.6663
}
```

**Observation:** Successfully retrieved comparison tables and discussion on the reactive-to-proactive shift from Pages 8-10.

---

## Query 3: Core Architectural Components

### Request
```http
POST /chat HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "question": "What are the key components of an AI agent?"
}
```

### Response
```json
{
  "answer": "The core pillars of agentic AI systems include perception, memory, reasoning/planning, and action/execution. Together, these allow the agent to interact with its environment, retain context, break down complex goals, and execute tasks using tools.",
  "retrieved_chunks": [
    {
      "page": 3,
      "score": 0.7473,
      "text": "Agentic AI\nAn Executive's Guide to In-depth\nUnderstanding of Agentic AI"
    },
    {
      "page": 17,
      "score": 0.7384,
      "text": "In this section, we explore the core components of agentic AI, explaining how they work together to enable autonomous decision-making."
    },
    {
      "page": 19,
      "score": 0.7048,
      "text": "2.1 The Core Pillars: From Perception to Execution\nAgentic AI systems function like a well-coordinated orchestra, with each component playing a vital role."
    },
    {
      "page": 22,
      "score": 0.6678,
      "text": "Tools, memory, and reasoning: the agent references past memory and selects tool integrations to interact with external APIs."
    }
  ],
  "confidence": 0.7473
}
```

**Observation:** Accurately extracted the Core Pillars (Perception to Execution) detailed in Section 2 of the eBook.

---

## Query 4: Enterprise Applications

### Request
```http
POST /chat HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "question": "What are some use cases of Agentic AI in enterprises?"
}
```

### Response
```json
{
  "answer": "According to the eBook, enterprise use cases include reducing review cycle times by 20% to 60%, automating complex multi-step workflows, improving forecasting and supply chain operations, and deploying multi-agent factory and e-commerce integrations.",
  "retrieved_chunks": [
    {
      "page": 5,
      "score": 0.7798,
      "text": "Table of Contents: 06 Practical Applications of Agentic AI, 05 Your Readiness for Agentic AI, 04 Orchestrating Agentic AI Systems."
    },
    {
      "page": 6,
      "score": 0.7212,
      "text": "\"Agentic AI going to be most of what we do in the future.\"\n\"Unlike simpler gen AI architectures, agents can produce high-quality content, reducing review cycle times by 20 to 60 percent.\""
    },
    {
      "page": 55,
      "score": 0.6841,
      "text": "Use cases: A North American manufacturer with 40 assembly lines adopting Factory Agentic Automation..."
    }
  ],
  "confidence": 0.7798
}
```

**Observation:** Retrieved direct McKinsey research citations and manufacturer case studies documented on Pages 6 and 55.

---

## Query 5: Multi-Agent Systems (MAS)

### Request
```http
POST /chat HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "question": "How do Multi-Agent Systems work?"
}
```

### Response
```json
{
  "answer": "Multi-Agent Systems (MAS) categorize agents into collaborative or specialized roles where tasks are distributed to the most suitable agents. By orchestrating individual agents, MAS optimizes collective intelligence and handles complex workflows that exceed the capacity of a single agent.",
  "retrieved_chunks": [
    {
      "page": 30,
      "score": 0.7832,
      "text": "What Are Multi-Agent Systems?\nAgentic systems can be categorized as single and multi-agent..."
    },
    {
      "page": 41,
      "score": 0.7314,
      "text": "assigned to the most suitable agents, optimizing the system's collective intelligence and performance. By executing tasks in parallel..."
    },
    {
      "page": 33,
      "score": 0.6920,
      "text": "3.2 MAS Scenario: A Supply Chain in Crisis\nA retail company relies on a global supply chain for sourcing, manufacturing, and shipping..."
    }
  ],
  "confidence": 0.7832
}
```

**Observation:** Retrieved Sections 3 and 4 covering Multi-Agent architecture and crisis scenarios.

---

## Query 6: Out-of-Scope Fallback (Strict Grounding Validation)

### Request
```http
POST /chat HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "question": "Who won the FIFA World Cup in 2018?"
}
```

### Response
```json
{
  "answer": "I couldn't find this in the Agentic AI eBook.",
  "retrieved_chunks": [
    {
      "page": 41,
      "score": 0.1096,
      "text": "assigned to the most suitable agents, optimizing the system's collective intelligence and performance..."
    },
    {
      "page": 59,
      "score": 0.1060,
      "text": "Aditya Vempaty is a Distinguished Research Scientist at Emergence AI, leading advanced research on Agentic AI..."
    },
    {
      "page": 48,
      "score": 0.0791,
      "text": "Organizational Maturity & Readiness assessment Framework..."
    },
    {
      "page": 55,
      "score": 0.0780,
      "text": "Use cases: A North American manufacturer with 40 assembly lines..."
    }
  ],
  "confidence": 0.1096
}
```

### Grounding Verification:
1. **Low Similarity Score:** The top retrieved chunk scored **0.1096**, which is significantly below the strict threshold of `0.30`.
2. **LangGraph Conditional Routing:** The `route` edge intercepted the low confidence score and bypassed LLM synthesis, directing the flow directly into the `fallback` node.
3. **Zero Hallucination:** The model refused outside knowledge (even though general LLMs know France won the 2018 World Cup), successfully upholding strict grounding to the knowledge base.
