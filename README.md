# AI-Powered RAG Workflow System

This workspace contains two complementary projects, **OrchestraRAG** and **Policy RAG Chat API**, designed to work together to deliver intelligent, Retrieval-Augmented Generation (RAG) workflows for enterprise-grade question answering and document retrieval.

---

## 📖 Projects Overview

Organizations today generate and manage vast amounts of unstructured data, from policy documents and compliance guidelines to internal knowledge bases. However, accessing this information quickly and accurately remains a significant challenge. The AI-Powered RAG Workflow System addresses this gap by combining advanced Retrieval-Augmented Generation (RAG) techniques with enterprise-ready orchestration and document retrieval capabilities.

This system is designed to empower enterprises with intelligent, context-aware tools for querying and managing their knowledge repositories. By leveraging state-of-the-art AI models and session-aware workflows, it enables users to retrieve precise, policy-grounded answers to their questions, complete with source citations. Whether it’s assisting employees with HR policies, ensuring compliance with regulatory standards, or streamlining access to organizational knowledge, this system delivers actionable insights in real-time.

The solution is built around two core components: OrchestraRAG, which orchestrates complex RAG workflows, and Policy RAG Chat API, which specializes in document retrieval and conversational AI for policy-related queries. Together, they provide a seamless, scalable, and secure platform for building intelligent chatbots, virtual assistants, and knowledge management systems.

With features like multi-session memory, real-time document updates, hybrid search capabilities, and conversation summarization, this system is tailored to meet the needs of modern enterprises. It not only saves time and improves accuracy but also enhances productivity by eliminating the friction of manual document searches. Designed with scalability and privacy in mind, it ensures compliance with enterprise security standards while delivering a superior user experience.

---

## 💡 Use Cases

The **AI-Powered RAG Workflow System** is designed to address a variety of enterprise needs. Below are some key use cases and examples of how the system can be applied:

| **Use Case**               | **Description**                                                                                     | **Example Queries**                                                                                     |
|-----------------------------|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| **HR Assistance**           | Empower employees to query HR policies and guidelines quickly and accurately.                      | - "What is the company's parental leave policy?"<br>- "How many vacation days am I entitled to?"        |
| **Compliance and Legal**    | Ensure compliance with regulatory standards by retrieving accurate answers to compliance-related queries. | - "What are the data retention policies for customer records?"<br>- "What are the GDPR compliance rules?"|
| **IT Support**              | Provide IT teams and employees with instant access to approved software lists and security guidelines. | - "What are the approved software tools for remote work?"<br>- "What is the company's password policy?" |
| **Knowledge Management**    | Centralize access to organizational knowledge, enabling employees to retrieve critical information efficiently. | - "Where can I find the latest company handbook?"<br>- "What are the steps for submitting an expense report?" |
| **Policy Updates**          | Allow administrators to update and manage policy documents in real-time, ensuring that users always have access to the latest information. | - "What are the recent changes to the remote work policy?"<br>- "When was the IT usage policy last updated?" |
| **Conversation Summarization** | Summarize chat sessions for compliance, analytics, or user recaps.                              | - "Summarize the last 5 questions asked in this session."<br>- "Provide a summary of the HR policy discussion." |
| **Multiple Questions Handling** | Handle workflows involving multiple related or unrelated queries within a single session.          | - "What is the company's remote work policy?"<br>- "What are the approved tools for remote work?"<br>- "How do I request additional software?" |

---

## ⚙️ Functional Features

### 1. [OrchestraRAG](OrchestraRAG/README.md)
OrchestraRAG is the orchestration layer that manages complex RAG workflows. It uses **LangGraph** to build stateful, multi-step pipelines and integrates with the **Policy RAG Chat API** for document retrieval and conversational context management.

### 2. [Policy RAG Chat API](policy-rag-api/README.md)
Policy RAG Chat API is a specialized RAG implementation for querying company policy documents. It provides document retrieval, session-aware memory, and summarization capabilities, which are leveraged by OrchestraRAG.

---

## 🏗️ Application Architecture

The following diagram illustrates how **OrchestraRAG** and **Policy RAG Chat API** work together to deliver intelligent, context-aware answers:

![Workflow](assets/RAGWorkflow.svg "Architecture Diagram of the AI-Powered RAG Workflow System") 

### Workflow Explanation:
1. **Client Interaction**: The client sends a query to the **OrchestraRAG** API.
2. **Orchestration**: OrchestraRAG processes the query using its LangGraph-based pipeline and determines whether to retrieve information from external RAG services or the **Policy RAG Chat API**.
3. **Policy Retrieval**: If the query pertains to company policies, OrchestraRAG forwards the request to the **Policy RAG Chat API**, which retrieves relevant document chunks from the knowledge base.
4. **Answer Generation**: The retrieved information is passed back to OrchestraRAG, which generates a context-aware response using an LLM.
5. **Response Delivery**: The final answer is returned to the client, along with source citations and optional conversation summaries.

---

## 🧠 Model Types Supported

The Policy RAG Chat API supports various types of AI models, each specialized for different tasks and resource requirements:

- **Small/Medium/Large Language Models**:  
  Choose between small, medium, and large foundation models based on your needs for speed, cost, and answer quality.  
  - *Small models* (e.g., `phi4`, `llama-3-8b`) offer fast responses and lower memory requirements, suitable for real-time interactions or resource-constrained environments.
  - *Large models* (e.g., `llama-3-70b`, `mistral-large`, `mixtral`) provide higher accuracy, more nuanced answers, and better handling of complex queries, at the cost of increased compute requirements and latency.
  - *Medium models* provide a balance between speed and quality.

- **Embedding Models**:  
  Specialized models (e.g., `mxbai-embed-large`, `bge-base`, `text-embedding-ada-002`) are used to convert text into numerical vectors for semantic search and retrieval. These models are optimized for generating dense representations that help the API find the most relevant policy sections for a user's query.

- **Evaluation Models**:  
  Separate models (often smaller/faster LLMs) are used to automatically assess the quality of generated answers for metrics like fluency, faithfulness, relevance, and conciseness. This allows for automated feedback and continuous improvement.

- **Summarization Models**:  
  For conversation or document summarization, the API can utilize models or chains optimized for condensing long text into concise summaries.

---

## 🔒 Scalability and Security

The **AI-Powered RAG Workflow System** is designed with enterprise scalability and security at its core, ensuring it meets the demands of modern organizations while safeguarding sensitive data.

---

## 🗂️ Workspace Structure

```text
.
├── OrchestraRAG/                # Orchestration layer for RAG workflows
├── policy-rag-api/              # Policy-focused RAG conversational API
└── README.md                    # Main documentation (this file)
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- `pip` for dependency management
- Graphviz (for workflow visualization in OrchestraRAG)

---

## 🛠️ Usage

### Running OrchestraRAG
```bash
cd OrchestraRAG
python -m api.main
```

### Running Policy RAG Chat API
```bash
cd policy-rag-api
python policy_rag_api.py
```

---

## 🤝 Community & Acknowledgments

We are deeply grateful to the open-source community and contributors who made this project possible. Below are some of the key tools, frameworks, and communities that supported our journey:

### **Libraries and Tools**
This project leverages the following open-source libraries and tools:
- **[LangGraph](https://www.langgraph.dev/)**: For building stateful, multi-step RAG workflows.
- **[LangChain](https://www.langchain.dev/)**: For document retrieval and conversational AI capabilities.
- **[FastAPI](https://fastapi.tiangolo.com/)**: For creating high-performance APIs.
- **[Redis](https://redis.io/)**: For session-aware memory and caching.
- **[Graphviz](https://graphviz.org/)**: For visualizing workflow pipelines.
- **[Python](https://www.python.org/)**: The core programming language used for development.

### **Inspiration**
This project was inspired by advancements in Retrieval-Augmented Generation (RAG) and the growing need for enterprise-grade AI solutions. Special thanks to the open-source community for their contributions to RAG research and tools.

### **Community Support**
We are grateful to the open-source communities and forums that provided guidance and support during the development of this project:
- **[LangChain Discord](https://discord.gg/langchain)** and **[LangChain GitHub Discussions](https://github.com/langchain-ai/langchain/discussions)** for retrieval-augmented generation (RAG) guidance.
- **[FastAPI Community](https://github.com/tiangolo/fastapi/discussions)** for framework support and best practices on the high-performance web framework.
- **[Ollama Community](https://discord.gg/ollama)** for help with self-hosted LLMs.
- **[Stack Overflow](https://stackoverflow.com/)** for troubleshooting a wide range of Python, API, and deployment questions.
- **[Redis Community](https://github.com/redis/redis)** for memory and persistence tips.
- **[ChromaDB Discussions](https://github.com/chroma-core/chroma/discussions)** for vectorstore advice and issue resolution.

---

Special thanks to all open-source contributors whose code and knowledge made this project possible! If you would like to contribute to this project, please feel free to submit a pull request or reach out to us with your ideas.