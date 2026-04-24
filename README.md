# Building a Chatbot with RAG from Scratch 🤖📚

> A Hands-On Guide to Retrieval-Augmented Generation, Agents, and Hallucination Prevention with LLMs

[**🌐 Workshop page**](https://hakanotal.github.io/workshop-rag-chatbot) - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MVo-hvo9jHgIz6E_adTpckZXqHU6jhR8?usp=sharing)

---

## 🚀 Quick Start

1. Open the [workshop page](https://hakanotal.github.io/workshop-rag-chatbot) or click the **Open in Colab** badge above
2. In Colab, go to **Runtime → Run all**
3. Wait ~2-3 minutes for setup
4. Follow along with the workshop, or explore on your own

**No local installation required.** Everything runs in Google Colab with a free account.

---

## 🎯 What you'll build

A full **Agentic RAG** pipeline over Mary Shelley's *Frankenstein*, end-to-end:

| # | Component | Tools |
|---|---|---|
| 0 | Local LLM setup | Ollama, Llama 3.2 (3B) |
| 1 | Plain LLM chat (baseline) | Ollama Python client |
| 2 | Document loading & chunking | LangChain |
| 3 | Embeddings & vector store | Nomic, ChromaDB |
| 4 | Simple RAG chain | LangChain LCEL |
| 5 | Agentic RAG (Retriever + Verifier) | CrewAI |
| 6 | Chat UI | Gradio |
| 7 | **Head-to-head comparison** | — |
| 8 | Evaluation (bonus) | Ragas |

**The finale:** we compare Plain LLM vs Simple RAG vs Agentic RAG on the same questions and watch how each handles:
- a classic hallucination trap *("What is the monster's name?")*
- specific factual lookups
- out-of-scope questions (grounding test)

---

## 📋 Prerequisites

- **A Google account** (for Colab)
- **A modern web browser**
- That's it. No Python install, no pip, no GPU required.

Optional but nice:
- A T4 GPU in Colab (Runtime → Change runtime type → T4 GPU) — gives you faster inference and lets you use a bigger model

---

## 📚 Key concepts covered

- **Why LLMs hallucinate** and why RAG is one of the cheapest, strongest defenses
- **Embeddings** — turning meaning into vectors
- **Chunking strategies** — fixed-length, sentence-based, recursive, semantic
- **Retrieval** — similarity search, top-k, vector databases
- **Prompt-based grounding** — the single most effective hallucination defense
- **Agentic workflows** — multi-agent systems for retrieval + verification
- **Evaluation** — faithfulness, answer relevancy, context precision/recall

---

## 🛠️ Tech stack

- **[Ollama](https://ollama.com)** — local LLM runtime
- **[LangChain](https://python.langchain.com)** — RAG primitives (loaders, splitters, retrievers)
- **[ChromaDB](https://www.trychroma.com)** — lightweight vector database
- **[CrewAI](https://www.crewai.com)** — multi-agent orchestration
- **[Gradio](https://www.gradio.app)** — chat UI in ~5 lines
- **[Ragas](https://docs.ragas.io)** — RAG evaluation metrics

All open source. All free.

---


## 📖 Recommended reading

- **RAG paper** — Lewis et al. 2020, [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401)
- **Ragas paper** — Es et al. 2023, [*Ragas: Automated Evaluation of Retrieval Augmented Generation*](https://arxiv.org/abs/2309.15217)
- **Self-RAG** — Asai et al. 2023, [*Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*](https://arxiv.org/abs/2310.11511)
- **Corrective RAG (CRAG)** — Yan et al. 2024, [*Corrective Retrieval Augmented Generation*](https://arxiv.org/abs/2401.15884)

---

## ⚖️ License

Released under the MIT License — use it, remix it, teach with it, build on it.

The *Frankenstein* text is in the public domain (Project Gutenberg eBook #84).