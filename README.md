
# 🧠 AI Lawyer Chatbot – RAG with LangChain + Groq + FAISS

This is an AI-powered legal assistant chatbot built using **LangChain**, **Grok (DeepSeek)** for inference, **FAISS** for vector storage, and **Sentence Transformers** for text embeddings. The chatbot uses Retrieval-Augmented Generation (RAG) to answer user queries based on uploaded legal PDFs.

---

## 🚀 Features

- ⚖️ **Legal Domain Focus** – Ask legal questions based on custom PDFs
- 🔍 **RAG Pipeline** – Combines LLM reasoning with document retrieval
- 🧠 **Grok/DeepSeek** – Fast and powerful inference using Groq API
- 🧩 **Sentence Transformers** – Converts text chunks into vector embeddings
- 🗂️ **FAISS Vector Store** – Efficient storage and retrieval of document chunks
- 📄 **PDF Support** – Upload and process legal documents

---

## 🛠️ Tech Stack

| Component            | Library / Tool                          |
|---------------------|------------------------------------------|
| LLM Inference        | `groq` (DeepSeek model)                 |
| Framework            | `LangChain`                             |
| Embeddings           | `sentence-transformers` (`MiniLM`)      |
| Vector Database      | `FAISS`                                 |
| Document Loader      | `PyPDFLoader`, `DirectoryLoader`        |
| Prompt Management    | `LangChain PromptTemplate`              |
| Environment Config   | `python-dotenv`                         |
| Language             | `Python 3.10+`                          |

---

## ⚙️ How It Works

1. **PDF Upload** – Upload your legal PDF documents.
2. **Text Chunking** – The text is split into manageable chunks using recursive character splitting.
3. **Vectorization** – Chunks are converted into dense vector embeddings using Sentence Transformers.
4. **Storage** – Embeddings are stored in a FAISS vector database.
5. **RAG Querying** – User inputs are embedded and relevant chunks are retrieved.
6. **LLM Inference** – The question and retrieved context are sent to Groq (DeepSeek) for response generation.

---

## 📦 Installation

```bash
git clone "https://github.com/Haseeb1511/AI_lawyer_chatbot.git"
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file with your API keys:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 🧪 Example Query

> “Can you explain the clause related to breach of contract in this document?”

---

## 📚 TODOs

- [ ] Add support for follow-up questions (chat history)
- [ ] Integrate multiple embedding models
- [ ] Add citation references to sources
- [ ] Dockerize the project

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
