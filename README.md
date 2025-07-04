# AI-Powered RAG Workflow System
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)]()
[![Docker](https://img.shields.io/badge/Containerized-Yes-green.svg)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-enabled-purple)](https://www.langgraph.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-✅-green)](https://fastapi.tiangolo.com/)

---

## ⚡ TL;DR (Quick Start)

```bash
# 1. Clone the repository
git clone https://github.com/vency-ai/RAG-Workflow-System.git
cd RAG-Workflow-System

# 2. Run everything with Docker
docker-compose up --build

# 3. Access the services
# OrchestraRAG API:       http://localhost:8000
# Policy RAG Chat API:    http://localhost:8011

This workspace contains two complementary projects, **OrchestraRAG** and **Policy RAG Chat API**, designed to work together to deliver intelligent, Retrieval-Augmented Generation (RAG) workflows for enterprise-grade question answering and document retrieval.

---

## 📖 Projects Overview

Organizations today generate and manage vast amounts of unstructured data, from policy documents and compliance guidelines to internal knowledge bases. However, accessing this information quickly and accurately remains a significant challenge. The AI-Powered RAG Workflow System addresses this gap by combining advanced Retrieval-Augmented Generation (RAG) techniques with enterprise-ready orchestration and document retrieval capabilities.

The AI-Powered RAG Workflow System empowers enterprises with intelligent, context-aware tools for querying and managing their knowledge repositories. By leveraging state-of-the-art AI models and session-aware workflows, it enables users to retrieve precise, policy-grounded answers to their questions, complete with source citations.

Whether assisting employees with HR policies, ensuring compliance with regulatory standards, or streamlining access to organizational knowledge, the platform delivers actionable insights in real-time.

Built for scale and privacy, the system includes features like multi-session memory, real-time document updates, hybrid search, and Conversation Summarization. These capabilities improve accuracy, save time, and enhance productivity by removing the friction of manual document searches.

Designed with enterprise security in mind, it ensures compliance with internal and external data protection requirements—without sacrificing performance or user experience.

The solution is built around two core components: OrchestraRAG, which orchestrates complex RAG workflows, and Policy RAG Chat API, which specializes in document retrieval and conversational AI for policy-related queries. Together, they provide a seamless, scalable, and secure platform for building intelligent chatbots, virtual assistants, and knowledge management systems.

With features like multi-session memory, real-time document updates, hybrid search capabilities, and Conversation Summarization, it is tailored to meet the needs of modern enterprises. It improves accuracy, saves time, and boosts productivity by eliminating the friction of manual document searches. Designed with scalability and privacy in mind, it ensures compliance with enterprise security standards while delivering a superior user experience.

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


Together, **OrchestraRAG** and **Policy RAG Chat API** provide the following capabilities:

### Intelligent Question Answering
- Retrieve relevant information from company policy documents using **policy-rag-api**.
- Generate context-aware, conversational answers using **OrchestraRAG** workflows.

### Stateful RAG Pipelines
- Build multi-step workflows using **LangGraph** to handle complex queries.
- Maintain conversational context across multiple questions using session-aware memory.

### Dynamic Document Retrieval
- Query and retrieve specific sections of policy documents in real-time.
- Support live updates to policy documents without downtime.

### Conversation Summarization
- Summarize chat sessions for compliance, analytics, or user recaps.
- Provide concise summaries of multi-step workflows.

### Answer Evaluation
- Assess the quality of responses based on relevance, fluency, and accuracy.
- Retry workflows with caching disabled if relevance falls below a threshold.

### Workflow Visualization
- Generate Graphviz diagrams to visualize RAG workflows for debugging and optimization.

### Extensibility
- Easily integrate with external APIs, databases, or messaging platforms (e.g., Slack, Teams).
- Customize prompts and workflows to suit specific business needs.





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

**Model selection is configurable per request or in the configuration file, allowing you to tailor the API's performance and behavior to your organization's needs.**

You can specify which model to use for answering, embedding, or evaluation by setting the `model` parameter in your API call, or by configuring defaults in `config.json` or `config.toml`.

---

## 🔒 Scalability and Security

The **AI-Powered RAG Workflow System** is designed with enterprise scalability and security at its core, ensuring it meets the demands of modern organizations while safeguarding sensitive data.

### **Scalability**
- **Modular Architecture**: The system is built with a modular design, allowing each component (e.g., OrchestraRAG, Policy RAG Chat API) to scale independently based on workload. This ensures optimal resource utilization and performance.
- **Redis-Backed Memory**: For session-aware workflows, the system leverages Redis for persistent memory storage, enabling horizontal scaling across multiple instances without losing conversational context.
- **Efficient Caching**: Frequently accessed queries and responses are cached to reduce latency and improve throughput, ensuring a seamless user experience even under high traffic.
- **Workflow Optimization**: The LangGraph-based orchestration in OrchestraRAG ensures efficient execution of multi-step workflows, minimizing processing overhead and maximizing response speed.

### **Security**
- **Completely Offline Deployment**: The system is designed to operate entirely offline, ensuring that no data is sent to external servers or the internet. This makes it ideal for air-gapped environments and organizations with strict data privacy requirements.
- **Self-Hosted Models**: All AI models and vector stores are hosted locally, ensuring that sensitive data never leaves the organization’s infrastructure.
- **Data Privacy Compliance**: The system adheres to enterprise-grade security standards, making it suitable for industries with stringent compliance requirements, such as healthcare, finance, and government.
- **Access Control**: API endpoints can be secured with authentication and role-based access control (RBAC) to ensure that only authorized users can interact with the system.
- **Audit Logging**: All queries, responses, and system actions are logged for auditing purposes, providing a transparent and traceable record of interactions.

### **Why Offline Deployment Matters**
In an era where data privacy and security are paramount, the ability to deploy a fully offline system is a significant advantage. By eliminating the need for internet access, the **AI-Powered RAG Workflow System** ensures:
- **Data Sovereignty**: All data remains within the organization’s infrastructure, reducing the risk of data breaches or unauthorized access.
- **Regulatory Compliance**: Offline deployment helps organizations comply with regulations like GDPR, HIPAA, and other data protection laws.
- **Operational Continuity**: The system remains fully functional even in environments with limited or no internet connectivity, ensuring uninterrupted access to critical information.

By combining scalability, security, and offline deployment, the **AI-Powered RAG Workflow System** provides a robust and reliable solution for enterprises looking to harness the power of AI while maintaining full control over their data.

---

## 🗂️ Workspace Structure


```text
.
├── OrchestraRAG/                # Orchestration layer for RAG workflows
│   ├── api/                     # FastAPI application and endpoints
│   ├── core/                    # LangGraph-based workflow logic
│   ├── config/                  # Configuration files
│   ├── tests/                   # Unit tests
│   ├── README.md                # Documentation for OrchestraRAG
│   └── requirements.txt         # Dependencies for OrchestraRAG
├── policy-rag-api/              # Policy-focused RAG conversational API
│   ├── cache_data/              # File-based response cache
│   ├── config/                  # Configuration files
│   ├── knowledgebase/           # Policy documents for retrieval
│   ├── logs/                    # Log files
│   ├── README.md                # Documentation for Policy RAG Chat API
│   └── requirements.txt         # Dependencies for Policy RAG Chat API
└── README.md                    # Main documentation (this file)
```

---
## 🔧 Tech Stack
- Python 3.11
- FastAPI
- Redis
- LangGraph + LangChain
- Ollama (for self-hosted LLMs)
- Graphviz (workflow rendering)
- Docker & Docker Compose

## 🚀 Installation

### Prerequisites
- Python 3.11+
- `pip` for dependency management
- Graphviz (for Workflow Visualization in OrchestraRAG)


### Setup Components Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/vency-ai/RAG-Workflow-System.git
   cd RAG-Workflow-System

   ```

2. Set up virtual environments for both projects:
   ```bash
   # For OrchestraRAG
   cd OrchestraRAG
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   pip install -r requirements.txt

   # For Policy RAG Chat API
   cd ../policy-rag-api
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Configure environment variables and settings:
   - Create `.env` files in both `OrchestraRAG` and `policy-rag-api` directories for secrets.
   - Modify `settings.toml` or `config.json` as needed.

### Container Deployment

To simplify the deployment process, all services in the **AI-Powered RAG Workflow System** can be containerized and managed using Docker. Follow the steps below to build and run the containers:

1. **Ensure Docker and Docker Compose Are Installed**:
   - Install [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) on your system.

2. **Build and Start All Services**:
   Navigate to the root directory of the project (where the `docker-compose.yml` file is located) and run:
   ```bash
   docker-compose up --build
   ```
3. Access the Services

- **OrchestraRAG**: Accessible at [http://localhost:8000](http://localhost:8000)  
- **Policy RAG Chat API**: Accessible at [http://localhost:8011](http://localhost:8011)  
- **Redis**: Runs on `localhost:6379` (used internally by the services)  
- **Ollama Server**: Runs on `localhost:11434` (used internally by the services)

---

4. Stop All Services

To stop and remove all running containers, use the following command:

```bash
docker-compose down
```
5. Persistent Data

- **Redis** data is stored in the `redis-data` volume.  
- **Ollama models** are stored in the `ollama-models` volume.  

These volumes ensure that data persists even after the containers are stopped.

By using **Docker**, you can easily deploy and manage all components of the system in a consistent and isolated environment.

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


## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with a detailed description.

---

## License

This workspace is licensed under the [MIT License](LICENSE).

---
## Future Enhancements

To further enhance the functionality and usability of the system, the following extensions are planned or can be implemented:

- **Web-Based Chat Interface**: Add a user-friendly web interface for real-time interaction with the RAG system.
- **Customizable Personas**: Allow users to select chatbot personas tailored to specific domains like HR or IT.
- **Slack or Microsoft Teams Integration**: Enable querying policies directly from workplace communication tools.
- **Advanced Analytics Dashboard**: Build an admin dashboard to monitor system usage, query trends, and answer quality.
- **Document Upload and Management Portal**: Create a portal for administrators to manage policy documents.
- **Voice Assistant Integration**: Integrate with Alexa, Google Assistant, or custom voice interfaces for hands-free interaction.


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

