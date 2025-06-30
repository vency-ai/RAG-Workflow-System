# Policy RAG Chat API

A robust Retrieval-Augmented Generation (RAG) conversational API for answering questions about about any company policy document—HR, IT, security, compliance, and more.Supports live policy updates, multi-session chat memory, and is future-proofed for multiple policy document support.

The RAG Chat API is a modern, retrieval-augmented conversational system. It combines private, self-hosted LLM capabilities (via Ollama), document retrieval, and per-session chat memory to provide context-aware, policy-grounded answers and summaries for Company policy documents.

It supports dynamic LLM model selection, live policy updates, conversation summarization, and detailed response metrics.

## ✨ Key Functional Features

- 💬 **Ask About Any Policy**  
  Employees and managers can ask questions in plain language about any company policy document. The system finds and returns answers straight from your uploaded policy files.

- 📂 **Supports Multiple Policy Documents**  
  Designed for flexibility, the system can handle multiple different policy documents (such as IT usage policy, security guidelines, code of conduct) either now or in future updates.

- 🔄 **Instant Policy Updates**  
  Administrators can update or replace policy documents at any time. The chatbot will immediately use the latest information for all future answers.

- 🧠 **Smart Chat Memory**  
  The chatbot remembers the conversation within a session, so follow-up questions and clarifications are answered more accurately and naturally.

- 🗄️ **Persistent Conversation Storage**  
  All active chats and conversation history are securely stored, so nothing is lost if the system restarts.

- ⬆️ **Easy Document Upload**  
  Admins can easily upload new or updated company policy documents through a simple web or API interface—no technical skills required.

- 📊 **Understands Complex Policy Structures**  
  The system can read and answer questions about tables and lists within policy documents.  
  *Example: "What are the approved software applications listed in our IT usage policy?"*

- 🦾 **Flexible AI Model Selection**  
  Choose from a variety of AI models to ensure the best balance of speed, cost, and accuracy for your organization’s needs.

- 📎 **Source Citations for Answers**  
  Every answer includes a reference to the exact section or page in the policy document where the information was found, making it easy to verify.

- 📝 **Conversation Summaries**  
  Quickly review the main points of a chat session with automatic conversation summaries.

- 📈 **Answer Quality Metrics**  
  Administrators can review and assess the quality of the chatbot’s responses, focusing on clarity, accuracy, relevance, and conciseness.

- ⚡ **Fast Response Caching**  
  Common questions and their answers are cached for rapid replies, keeping the experience fast and efficient for everyone.

- 📑 **Advanced File Management & Metadata Filtering**  
  Easily preview, download, or extract metadata from any policy document via dedicated API endpoints. List all available document metadata keys and values to build robust metadata filters for retrieval.

---

**Policy RAG API** is an intelligent conversational system designed to answer questions about any company policy document—including HR, IT, security, compliance, and more. It helps employees, managers, and administrators quickly find accurate information directly from official documents, ensuring clarity and compliance across the organization.

---
## 📋 Feature Support Overview

The Policy RAG Chat API is a modern, retrieval-augmented conversational system. It combines private, self-hosted LLM capabilities (via Ollama), document retrieval, and per-session chat memory to provide context-aware, policy-grounded answers and summaries for Company policy documents. Below, we detail the system’s support for advanced features such as reasoning/chaining, privacy, integrations, customization, offline operation, and more.

---

## 🧮 Feature Support Matrix

| Feature                        | Supported | Description |
|--------------------------------|:---------:|-------------|
| **Reasoning and Chaining**     | ✅ Yes    | The system leverages LangChain to combine document retrieval (semantic search over your Company policy) and LLM-based reasoning. Each API call builds a chain that both finds relevant Company policy sections and uses an LLM to synthesize a context-aware answer, supporting multi-step reasoning and contextual conversation. |
| **External APIs and AI Models**| ✅ Yes    | The app integrates with external/local APIs for LLM inference (e.g., Ollama) and for generating embeddings (e.g., local or remote embedding models). These are called over HTTP and can be swapped for other providers as needed. All retrieval and generation uses these modular external AI endpoints. |
| **Custom Prompt**              | ✅ Yes    | Prompts are fully configurable via the `config.json` file under the `"prompts"` array. You can define multiple named templates (e.g., "strict_policy", "friendly_hr"), control their content, and set a default. Prompts can use context, chat history, and question variables, and new templates can be added without code changes. |
| **Context-Awareness**          | ✅ Yes    | The system provides deep context-awareness via two mechanisms: (1) Per-session chat memory, so each session maintains a running conversation, and (2) Retrieval-augmented generation (RAG), which fetches the most relevant Company policy sections for each question and supplies them as LLM context. This ensures answers are both conversational and policy-grounded. |
| **Offline Operations**         | ✅ Yes    | The entire stack (LLM via Ollama, embeddings, vectorstore, Redis) can be run on-premise or in an air-gapped environment, enabling fully offline operation. No component requires internet access if you self-host all dependencies and models. |
| **Self-Hosted Models**         | ✅ Yes    | The platform is designed to work with self-hosted LLMs (e.g. via Ollama), local vectorstores (Chroma/FAISS), and local Redis instances. No reliance on cloud LLM providers unless you configure it that way; all AI/ML can be run in your own infrastructure. |
| **Private and Protected Data** | 🟡 Partial| All data (policy markdown, chat histories) are accessed through the API and are not publicly exposed. However, there is no built-in authentication, authorization, or encryption; data privacy is reliant on how you deploy the API. For stricter privacy or regulatory compliance, you should add authentication, access control, and (if needed) encryption at rest and in transit. |
| **Integrations**               | 🟡 Partial| The core system integrates with Redis (for persistent conversation memory), with Ollama-compatible LLMs, and supports file-based caching. Ready-made integrations for messaging/communication platforms (Slack, Teams, etc.) or other business tools are not included, but can be built on top of the API. |
| **Chain Routing**              | 🟡 Partial| The API allows per-request model overrides (dynamic LLM selection), but does not implement a full chain router (automatic routing to different chains based on intent or query type). It is extensible to support chain routing by inspecting input and dispatching to different chains as needed. |
| **Personas**                   | 🟡 Partial| Multiple prompt templates allow you to set different "personas" or answer styles (e.g., strict, friendly). However, there is no built-in user-side dynamic persona switching; persona is determined by prompt selection in config, or could be extended to be per-session or per-request. |
| **Moderation**                 | ❌ No     | There is no built-in moderation (no toxicity/profanity detection, compliance checks, or abuse filtering). All queries and responses are processed as-is. If you require moderation, you should integrate third-party moderation APIs or implement rules-based filtering at the application or gateway level. |
| **Evaluation of Answers**      | ✅ Yes    | Built-in LLM-based evaluation of answers (fluency, faithfulness, relevance, conciseness), with scores and settings configurable in config. |
| **Evaluation Management**      | ✅ Yes    | `/eval` endpoint to evaluate answers, `/eval_clear` to clear all evaluations, and evaluation results are attached to cache entries. |
| **Cache Usage Prediction**     | ✅ Yes    | `/will_use_cache` endpoint reports if a cache hit would occur for a given payload, matching `/invoke` logic. |
| **Advanced File Management & Metadata Filtering** | ✅ Yes | Preview, download, and extract metadata from policy documents. List all available frontmatter keys and values to facilitate rich metadata-based document retrieval and filtering. |

---

**Legend:**  
✅ Yes — Fully supported  
🟡 Partial — Some support; see description  
❌ No — Not currently supported

---
## 🧩 LangChain Features Used

This project leverages several core LangChain features for building the RAG pipeline:

| LangChain Feature                    | Purpose in This Project                                                  |
|--------------------------------------|--------------------------------------------------------------------------|
| **Document Loaders**                 | Load and parse the Policy markdown file, including handling markdown tables.|
| **Text Splitters**                   | Chunk the loaded markdown document (while preserving tables) into pieces suitable for semantic search and LLM context windows.|
| **Embeddings**                       | Generate vector representations for each document chunk using a local embedding model (Ollama or compatible).|
| **Vectorstores (Chroma/FAISS)**      | Store and efficiently search for semantically similar chunks to the user query.|
| **Retrievers**                       | Fetch the top-k most relevant document chunks to supply as context for the LLM.|
| **LLM Chains**                       | Construct a prompt (including user question and retrieved context) and route it through the selected LLM for answer generation.|
| **Chat Memory**                      | Maintain conversation history per session, using either in-memory or Redis-backed storage for continuity and context.|
| **Prompt Templates**                 | Custom prompt templates are defined in `config.json` and injected into the RAG chain. The default prompt template is selected via a config flag and controls how the context, chat history, and question are presented to the LLM.|
| **Callbacks/Instrumentation**        | Configurable LangChain callback instrumentation/logging, controlled by the `langchain_logging` section in config. Tracks timings and token counts for monitoring and debugging.   |
| **Conversation Summarization**       | Uses LangChain's summarization chains (`load_summarize_chain`) to summarize the last N turns, or the entire chat, on demand. The `/summarize_session` API endpoint exposes this, allowing you to specify both the summarization chain type (e.g. "stuff", "map_reduce") and how many recent messages to summarize.|

**Purpose:**  
LangChain orchestrates the end-to-end RAG flow: from document ingestion, chunking, retrieval, and context assembly to chat memory management, LLM invocation, and now conversation summarization—enabling modular, production-grade question answering and session analytics over your Company policy.

---

## 🏗️ Application Architecture

```text
RAG-API/
├── cache_data/
│   └── response_cache.jsonl         # File-based cache of all responses (with evaluations)
├── config/
│   └── config.json                  # Main configuration (now includes evaluation section)
├── knowledgebase/
│   └── policy-1278.md                # Your Company policy markdown file
├── logs/
│   └── rag_policy.log
├── policy_rag_chain.py              # Core logic for document  processing and retrieval-augmented generation (RAG) chains
│                                      to support question answering over any company policy documents.
├── policy_rag_api.py                # FastAPI application and API endpoints for the Policy RAG system, supporting 
│                                      conversational question answering over any company policy documents.
├── evaluate_rag_cache.py            # Evaluation logic and helpers
└── README.md
```

---

## 📝 Example `config.toml`

```toml
[data]
kb_dir = "knowledgebase"
markdown_file = "policy-1278.md"

[embedding]
embedding_model = "mxbai-embed-large:latest"
vectorstore_dir = "chroma_db"

[llm]
ollama_base_url = "http://127.0.0.1:11434"
llm_model_name = "phi4"

[[prompts]]
name = "strict_policy"
default = true
template = """You are a helpful HR assistant. Only use the context from the Company policy. If unsure, say: 'I do not know based on the provided Company policy.'"""
```
---
### **Cache Section in Detail**

| Key              | Example Value             | Description                                                                 |
|------------------|--------------------------|-----------------------------------------------------------------------------|
| `cache_dir`      | `"cache_data"`           | Directory for cache file (created if missing).                              |
| `cache_file`     | `"response_cache.jsonl"` | JSONL file storing all cached responses (one JSON object per line).         |
| `file_cache_ttl` | `604800`                 | TTL for cache entries, in seconds (default 1 week = 604,800 seconds).       |

- **Why JSONL?**  
  Each line is a full response object (with timestamp and cache key), allowing for efficient appending and lookup.
- **How TTL Works:**  
  When a cached answer is older than `file_cache_ttl`, it is ignored and recomputed on next query.

---

### **Evaluation Section in Detail**

| Key              | Example Value             | Description                                                                 |
|------------------|--------------------------|-----------------------------------------------------------------------------|
| `model`          | `"phi4"`                 | Model used for answer evaluation.                                           |
| `ollama_base_url`| `"http://localhost:11434"` | Ollama endpoint for evaluation LLM.                                       |
| `cache_dir`      | `"cache_data"`           | Directory for evaluated cache file.                                         |
| `cache_file`     | `"response_cache.jsonl"` | File where evaluated cache entries are stored.                              |
| `prompts`        | `{...}`                  | Customizable prompts for scoring fluency, faithfulness, relevance, conciseness. |

- **How Evaluation Works**:  
  Each answer can be scored by the evaluation LLM for four metrics. All scores are attached to the cache entry.

---

## 🧑‍💻 Usage

### 1. **Start Redis (optional, for persistent memory)**
```sh
redis-server
```

### 2. **Start Ollama**
```sh
ollama serve
```

### 3. **Start the API**
```sh
cd RAG-API
uvicorn policy_rag_api:app --host 0.0.0.0 --port 8011
```

### 4. **Query the API with session support, model override, and cache options**

**Default (response is always saved to cache):**
```sh
curl -X POST "http://localhost:8011/invoke" \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: user42" \
  -d '{"input": {"question": "What is the Company policy for new hires?"}}'
```

**Use the cache if available (will serve from cache for repeated identical queries/model):**
```sh
curl -X POST "http://localhost:8011/invoke" \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: user42" \
  -d '{"input": {"question": "Show me the leave policy for employee", "model": "llama-3", "use_cache": true}}'
```

**Force fresh answer, but still save it to cache for next time:**
```sh
curl -X POST "http://localhost:8011/invoke" \
  -H "Content-Type: application/json" \
  -d '{"input": {"question": "What are the mandatory training for new employee?", "use_cache": false}}'
```

---

### 5. **Predict Cache Usage with `/will_use_cache`**

Check if a given payload would use the cache (without invoking the LLM):

```sh
curl -X POST "http://localhost:8011/will_use_cache" \
  -H "Content-Type: application/json" \
  -d '{"input": {"question": "What is the Company policy for new hires?", "use_cache": true}}'
```

**Response Example:**
```json
{
  "will_use_cache": true,
  "reason": "Valid, unexpired cache entry found and will be used.",
  "cache_key": "phi4::what is the Company policy for new hires?",
  "cached_at": 1720000000
}
```
or
```json
{
  "will_use_cache": false,
  "reason": "No cached entry found for this question/model.",
  "cache_key": "phi4::what is the Company policy for new hires?"
}
```

---
### 6. **/invoke Response Format**

Includes timing, token usage, model, chat history, sources, and a calculation note.

```json
{
  "result": {
    "question": "How do I request IT support for a software installation?",
    "chat_history": [
      {"type": "human", "content": "How do I request IT support for a software installation?"},
      {"type": "ai", "content": "To request IT support for software installation, submit a ticket via the IT Service Portal and include the software name and your device information."}
    ],
    "model": "llama-3",
    "answer": "To request IT support for software installation, submit a ticket via the IT Service Portal and include the software name and your device information.",
    "sources": [
      {
        "text": "...chunk of IT support policy text...",
        "metadata": { "section": "Submitting IT Support Requests" },
        "distance": 0.01
      }
    ],
    "time_summary": {
      "retrieval_time": 0.032,
      "llm_time": 0.489,
      "total_time": 0.532
    },
    "token_summary": {
      "query_tokens": 13,
      "context_tokens": 92,
      "response_tokens": 47
    },
    "note": "Token counts are estimated using tiktoken's cl100k_base encoding."
  }
}
```

- The `note` field will indicate whether token counts are exact (OpenAI), estimated (tiktoken), or rough (fallback).
- If tiktoken is not installed, a fallback estimate is used.

---

### 6a. **Answer Evaluation During /invoke**

**Automatic Answer Evaluation**

If you add `"eval": true` to your `/invoke` payload:

- The answer will be automatically evaluated by an LLM (using the model specified in your config) for:
  - **Fluency**
  - **Faithfulness**
  - **Relevance**
  - **Conciseness**
- The evaluation scores will be included in the API response (top-level `"evaluation"` field) and attached to the cache entry.
- If a cached answer exists, the system compares the evaluation scores and may return the higher-rated answer.

**Example payload:**
```json
{
  "input": {
    "question": "What is the PTO policy?",
    "model": "llama3.2",
    "use_cache": false,
    "metadata_filter": { "type": "contractor_policy" },
    "eval": true
  }
}
```

**Example response (with evaluation):**
```json
{
  "result": {
    ...
  },
  "evaluation": {
    "fluency": 4,
    "faithfulness": 3,
    "relevance": 4,
    "conciseness": 3
  },
  "answer_source": "cache",
  "decision": "Cached answer used because its relevance score (4) >= fresh (3)"
}
```
- If you do **not** see the `"evaluation"` field in your response, make sure you are using `"eval": true` (not `"eval_now": true`) in the payload, and that your backend supports evaluation logic.

---

### 7. **Update Company policy on the Fly**

Upload a new markdown file and clear vectorstore cache and file cache (affects all sessions):

```sh
curl -X POST "http://localhost:8011/update-policy" \
  -F "file=@/path/to/your/new_policy.md"
```
After updating, the next `/invoke` call will use the new Company policy contents.  
*The file cache is also cleared automatically when the policy is updated.*

---

### 8. **Manually Clear All Cached Responses**

To force a cache clear (e.g., after a manual config change):

```sh
curl -X POST "http://localhost:8011/refresh_cache"
```

---
### 9. **Summarize the Conversation with /summarize_session**

Get a summary of the last N messages (or the entire session) using your choice of summarization chain type:

```sh
curl -X POST "http://localhost:8011/summarize_session" \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: user42" \
  -d '{"chain_type": "stuff", "last_n": 6}'
```

- **chain_type**: `"stuff"` for short chats, `"map_reduce"` for longer or more detailed conversations.
- **last_n**: The number of most recent message-pairs to summarize. Omit to summarize the entire chat.

**Response Example:**
```json
{
  "summary": "The employee asked about the process for submitting an IT support ticket and how to check ticket status, and received step-by-step instructions from the IT policy."
}
```

Use this for analytics, UI recaps, or to provide users a concise summary of the conversation so far!

---

### 10. **Manually Reload the Vectorstore with /reload-data**

After updating your Company policy markdown, or at any time, you can eagerly rebuild the vectorstore and view chunking statistics:

```sh
curl -X POST http://localhost:8011/reload-data
```

**Response Example:**
```json
{
    "message": "Company policy data reloaded and vectorstore rebuilt.",
    "session_id": "default-session",
    "model": "phi4",
    "number_of_chunks": 42,
    "min_chunk_size": 120,
    "avg_chunk_size": 786.2,
    "max_chunk_size": 1002,
    "log": [
        "Loaded markdown file: knowledgebase/policy-1278.md",
        "Split into 42 chunks (min/avg/max: 120/786.2/1002)",
        "Rebuilt vectorstore for session_id: default-session and model: phi4"
    ]
}
```
This is especially useful after a policy update, to force immediate chunking/vectorstore rebuild and to verify chunking statistics.

---

### 11. **Clear the Chat History for a Specific Session**

You can clear the chat memory for a session (e.g., to start fresh or for privacy) using:

```sh
curl -X POST "http://localhost:8011/clear_session_history" \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: my-session-abc123"
```

**Response Example:**
```json
{
  "message": "Chat history cleared for session_id: my-session-abc123",
  "active_chain_sessions": 3
}
```

- This clears the chat history for only the session specified by `X-Session-Id`.
- The response shows how many session/model chains are currently active in memory.

---

### 12. **Evaluate All Answers and View Results**

**Run evaluation on all answers in the cache:**
```sh
curl -X POST "http://localhost:8011/eval_all"
```
**Response Example:**
```json
{
  "message": "Evaluation completed.",
  "total_entries": 3,
  "evaluated": 3,
  "skipped": 0,
  "errors": 0,
  "details": [
    {
      "key": "llama3.2::what is the Company policy for paid parental leave?",
      "answer": "Employees are eligible for up to 12 weeks of paid parental leave as described in the HR policy.",
      "fluency": 4,
      "faithfulness": 3,
      "relevance": 3,
      "conciseness": 3
    }
    // ... up to 25 entries
  ]
}
```
- Each answer is scored for fluency, faithfulness, relevance, and conciseness by the evaluation LLM, and results are attached to each cache entry.

**Clear all evaluation results:**
```sh
curl -X POST "http://localhost:8011/eval_clear"
```
**Response Example:**
```json
{
  "message": "Evaluations cleared.",
  "total_entries": 3,
  "cleared": 3,
  "errors": 0
}
```

- This removes all evaluation fields from all cache entries, allowing you to re-run evaluation or reset metrics.

---

### 13. **Advanced File Management Endpoints**

#### **Preview or Download a Markdown File**
```sh
curl -X GET "http://localhost:8011/get-doc?filename=YOUR_FILE.md"
```
Replace `YOUR_FILE.md` with the actual filename.

#### **Get Metadata (YAML Frontmatter) of a Markdown File**
```sh
curl -X GET "http://localhost:8011/doc-metadata?filename=YOUR_FILE.md"
```
Replace `YOUR_FILE.md` with your desired filename.

#### **List All Unique Metadata Keys/Values Across All Documents**
```sh
curl -X GET "http://localhost:8011/list-metadata"
```

These endpoints allow you to inspect, preview, and query document metadata—enabling powerful metadata-based retrieval and management.

---

## 🛡️ Best Practices

- **Always pass a unique `X-Session-Id`** for each user or UI session to maintain chat context.
- **Let users specify a model** in each `/invoke` call if you want to offer model flexibility.
- **Validate your markdown**: Tables and headers should be well-formed for best chunking and retrieval.
- **Monitor logs**: The app logs all errors, model changes, timing, token logic, and major steps to `logs/rag_policy.log`.
- **Deploy Redis in production** to persist chat memory and allow horizontal scaling.
- **Update LLM and embedding models as needed** for your data and accuracy needs.
- **Graceful Degradation**: System falls back to in-memory chat if Redis is offline.
- **Adjust cache TTL** in config as needed (`file_cache_ttl`).
- **Leverage Conversation Summarization** for compliance, analytics, or to show users a running recap of their session.
- **Use `/reload-data` after updating the Company policy file** to force chunking/vectorstore rebuild and view chunk stats.
- **Use `/refresh_cache` at any time** to clear the file-based response cache.
- **Use `/clear_session_history` to reset chat memory for a session.
- **Use `/eval` and `/eval_clear` for managing answer evaluation metrics.
- **Use `/will_use_cache` to check/predict cache hits before invoking expensive LLM calls.
- **Use new file management endpoints to preview, download, or filter documents by metadata for precise retrieval.**

---

## 🔧 Extending the Application

See the [previous suggestions](#) for adding hybrid search, advanced memory, summarization, source citations, streaming, and more using LangChain!

---

## 🏁 Conclusion

This application is a production-ready, RAG-powered API for Company policy chatbots, knowledge bases, and HR assistants.  
It features robust document handling, session-aware memory, model flexibility, live policy updates, conversation summarization, manual vectorstore reload, efficient file-based response caching, per-session chat resets, and built-in answer evaluation and cache prediction—ready for enterprise deployment.

---

**Questions? PRs welcome!**