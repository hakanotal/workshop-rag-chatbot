# Workshop Plan v2 — Building a Chatbot with RAG from Scratch
**Presenter:** Hakan Tugrul Otal · **Duration:** 60–75 min · **Format:** 15 min talk + 45–60 min hands-on Colab notebook
**Project path:** `/Users/hakan/Desktop/NTIR Workshop`
**Sample document:** *Frankenstein* by Mary Shelley (Project Gutenberg eBook #84)

---

## Key decisions locked in
- ✅ **Google Colab** (free tier w/ T4 GPU) — no local install required for participants
- ✅ **Ollama on Colab** — keeps the "run your own LLM" pedagogical thread; same code runs locally later
- ✅ **Frankenstein** as the sample document (famous, public domain worldwide, built-in hallucination demo)
- ✅ **2-agent Agentic RAG**: Retriever + Verifier
- ✅ **Comparison finale**: No-RAG vs Simple RAG vs Agentic RAG head-to-head

---

## 1. Revised Time Budget

| Block | Time | What happens |
|---|---|---|
| Welcome + open Colab link | 3 min | Everyone clicks the link, runs setup cell |
| Concept presentation (slides) | 15 min | Hallucination problem → RAG → Agentic RAG |
| Notebook §1: Colab + Ollama setup | 4 min | Run setup cell, pull models (happens while you explain) |
| Notebook §2: Simple chatbot | 5 min | First LLM call, show hallucination on Frankenstein |
| Notebook §3: Load + chunk the book | 7 min | PyPDFLoader, RecursiveCharacterTextSplitter |
| Notebook §4: Embed + ChromaDB | 6 min | Nomic embeddings, similarity search |
| Notebook §5: Simple RAG chain | 7 min | LCEL chain, strict grounding prompt |
| Notebook §6: Agentic RAG (CrewAI) | 10 min | Retriever + Verifier agents |
| Notebook §7: Gradio UI | 4 min | Chat interface, public share link |
| Notebook §8: **Head-to-head comparison** | 6 min | 5 questions × 3 systems, side-by-side table |
| Q&A + buffer | 5–8 min | |
| **Total** | **72–75 min** | |

**Cuttable in emergency:** §7 Gradio (5 min saved), Agentic Verifier loop complexity (3 min), §8 reduced to 3 questions (3 min).

---

## 2. Project Folder Structure

Set up at `/Users/hakan/Desktop/NTIR Workshop`:

```
NTIR Workshop/
├── index.html                   # GitHub Pages landing page (workshop entry point)
├── rag_workshop.ipynb           # Main notebook (Colab-ready)
├── workshop_slides.pptx         # Presentation deck
├── README.md                    # GitHub repo front page
└── workshop_plan_v2.md          # This planning doc (presenter-only)
```

The repo lives at **github.com/hakanotal/workshop-rag-chatbot**, with GitHub Pages serving `index.html` from the root. Participants open the page, click **Open notebook in Colab**, and they're in.

*Frankenstein* is downloaded by the notebook itself at runtime, so we don't commit it to the repo.

---

## 3. Action Items (Revised)

### This weekend (7 days out)
- [ ] Push the repo contents to `github.com/hakanotal/workshop-rag-chatbot`
- [ ] Enable GitHub Pages (Settings → Pages → Deploy from main branch, root)
- [ ] Verify the landing page renders at `hakanotal.github.io/workshop-rag-chatbot`
- [ ] Build the notebook end-to-end in Colab. Time every cell. Save outputs.
- [ ] Test on a **fresh** Colab runtime with GPU disabled first, then enabled — both should work.
- [ ] Verify Ollama install on Colab works reliably (sometimes flaky; have Plan B ready)

### 5 days out
- [ ] Dry-run the full workshop, on the clock, in front of a mirror or a lab-mate
- [ ] Prepare a fallback notebook using **Groq API** (free, fast, no Ollama needed) in case Ollama-on-Colab has issues on workshop day

### 2 days out
- [ ] Final slide polish
- [ ] Rehearse the money-demo moments: hallucination on "monster's name," simple RAG answer, verifier catching an error
- [ ] Test the landing page + Colab link on a phone browser (some attendees may use tablets)

### Day of
- [ ] Arrive 30 min early
- [ ] Open the Colab notebook early so the runtime is warm
- [ ] Have the landing-page URL as a QR code on the first slide
- [ ] Have backup Groq-API notebook tab ready

---

## 4. Slide Deck (15 min, ~12 slides)

*Unchanged from v1 except:*
- **Slide 9 (Our Tools)** — drop "local laptop" framing, say "Colab + Ollama"
- **Slide 11** — mention the final comparison demo explicitly as the payoff
- **Slide 12** — QR code goes to the GitHub Pages landing page (`hakanotal.github.io/workshop-rag-chatbot`); the page has an **Open notebook in Colab** button as the primary CTA

See v1 for full slide-by-slide content. Core flow unchanged:
1. Title | 2. About me + agenda | 3. Hallucination problem | 4. Three ways to teach an LLM | 5. RAG pipeline diagram | 6. Embeddings | 7. Chunking | 8. Agentic RAG | 9. Our stack | 10. Eval + hallucination prevention | 11. What we'll build | 12. Go

---

## 5. Notebook Structure (Colab-Ready)

### §0 — Setup (runs once, ~2-3 min on Colab)
```python
# Install Ollama in Colab
!curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server in background
import subprocess, time
subprocess.Popen(['ollama', 'serve'])
time.sleep(3)

# Pull models
!ollama pull llama3.2:3b       # ~2GB, good quality, fine on Colab CPU
!ollama pull nomic-embed-text  # ~270MB embedding model

# Install Python dependencies
!pip install -q langchain langchain-community langchain-core \
               chromadb crewai crewai-tools ollama gradio pypdf ragas

# Download Frankenstein
!wget -q https://www.gutenberg.org/files/84/84-0.txt -O frankenstein.txt
print("Setup complete ✓")
```

**Pedagogical note for you:** kick this cell off *before* you start the slides. It'll be done by the time participants arrive at §0.

### §1 — Talk to the LLM (no RAG yet)
```python
import ollama
resp = ollama.chat(model='llama3.2:3b', messages=[
    {'role': 'user', 'content': 'What is the name of the monster in Frankenstein?'}
])
print(resp['message']['content'])
```
**Money moment:** the small LLM will likely say "Frankenstein" or confidently give a wrong answer. You now have the motivation for the rest of the workshop. Let this land.

### §2 — Load and chunk the book
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

with open('frankenstein.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# Strip Project Gutenberg header/footer (simple heuristic: find 'Letter 1' start)
start = text.find("Letter 1")
end = text.find("*** END OF THE PROJECT GUTENBERG")
book = text[start:end]

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = [Document(page_content=c) for c in splitter.split_text(book)]
print(f"Book length: {len(book):,} chars → {len(chunks)} chunks")
# Show a sample chunk
print("\n--- Sample chunk ---\n", chunks[10].page_content[:400])
```

### §3 — Embed + ChromaDB
```python
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OllamaEmbeddings(model='nomic-embed-text')
vectorstore = Chroma.from_documents(chunks, embeddings)

# Test retrieval
results = vectorstore.similarity_search("Does the monster have a name?", k=3)
for i, r in enumerate(results):
    print(f"--- Match {i+1} ---\n{r.page_content[:300]}\n")
```

### §4 — Simple RAG chain
```python
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOllama(model='llama3.2:3b')
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_template("""
You are a literary assistant answering questions about the novel Frankenstein.
Use ONLY the context below. If the answer is not in the context, say "The text does not specify."

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# The money demo — same question, now grounded
print(rag_chain.invoke("Does the monster have a name?"))
```

### §5 — Agentic RAG (2 agents: Retriever + Verifier)
```python
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

@tool("search_frankenstein")
def search_tool(query: str) -> str:
    """Searches the novel Frankenstein for relevant passages."""
    docs = vectorstore.similarity_search(query, k=4)
    return "\n\n---\n\n".join(d.page_content for d in docs)

retriever_agent = Agent(
    role="Literary Researcher",
    goal="Find the most relevant passages from Frankenstein and draft a grounded answer with citations.",
    backstory="An expert in 19th century literature who always cites the text.",
    tools=[search_tool],
    llm=llm,
    verbose=True,
)

verifier_agent = Agent(
    role="Fact Verifier",
    goal="Check that the answer is fully supported by the retrieved passages. Flag any claims not grounded in the text.",
    backstory="A skeptical editor who demands textual evidence for every claim.",
    llm=llm,
    verbose=True,
)

def run_agentic_rag(question: str) -> str:
    retrieve_task = Task(
        description=f"Answer this question using the search_frankenstein tool: {question}",
        expected_output="A draft answer with supporting quotes.",
        agent=retriever_agent,
    )
    verify_task = Task(
        description="Review the draft answer. If any claim is not supported by the retrieved passages, correct it or say 'The text does not specify.'",
        expected_output="A verified, grounded final answer.",
        agent=verifier_agent,
        context=[retrieve_task],
    )
    crew = Crew(
        agents=[retriever_agent, verifier_agent],
        tasks=[retrieve_task, verify_task],
        process=Process.sequential,
    )
    return crew.kickoff(inputs={"question": question})

print(run_agentic_rag("Does the monster have a name?"))
```

### §6 — Gradio UI
```python
import gradio as gr

def chat_fn(message, history):
    return rag_chain.invoke(message)

gr.ChatInterface(
    chat_fn,
    title="Ask Frankenstein",
    description="A RAG-powered chatbot trained on Mary Shelley's novel.",
    examples=["Who rescues Victor from the Arctic ice?",
              "Why does the monster kill William?",
              "Does the monster have a name?"],
).launch(share=True)
```

### §7 — HEAD-TO-HEAD COMPARISON (the finale)

This is the payoff. Show a table comparing three approaches on the same questions:

```python
import pandas as pd
from IPython.display import display, HTML

# Bare LLM (no context)
def plain_llm(q):
    return ollama.chat(model='llama3.2:3b', messages=[
        {'role': 'user', 'content': q}])['message']['content']

# Simple RAG
def simple_rag(q):
    return rag_chain.invoke(q)

# Agentic RAG
def agentic_rag(q):
    return str(run_agentic_rag(q))

test_questions = [
    "Does the monster have a name?",
    "Who is the captain of the ship that rescues Victor at the end?",
    "What does the monster read that shapes his understanding of humanity?",
    "What is the capital of Switzerland?",  # Not in the book — tests grounding
    "How does Victor Frankenstein bring the creature to life?",
]

results = []
for q in test_questions:
    results.append({
        "Question": q,
        "Plain LLM": plain_llm(q)[:200],
        "Simple RAG": simple_rag(q)[:200],
        "Agentic RAG": agentic_rag(q)[:200],
    })

df = pd.DataFrame(results)
display(HTML(df.to_html(escape=False)))
```

**Expected findings to highlight:**

| Question | Plain LLM likely says | Simple RAG | Agentic RAG |
|---|---|---|---|
| Monster's name? | "Frankenstein" (wrong) | "The monster has no name" | Same + cites passage |
| Ship captain? | Maybe "Walton" (right) or hallucinates | "Robert Walton" + context | Same + verified |
| What does monster read? | Probably wrong specifics | "Paradise Lost, Plutarch's Lives, Sorrows of Werter" | Same + verified |
| Capital of Switzerland? | "Bern" (right, but irrelevant!) | "The text does not specify" | "Not in corpus" |
| How is creature made? | Vague Hollywood version | Grounded in actual gothic-science passages | Same + verified |

**Key discussion points:**
- Plain LLM: confident but often wrong on specifics; can answer off-topic Qs it shouldn't
- Simple RAG: grounded but brittle (depends on chunk quality + retrieval)
- Agentic RAG: slower, more robust, catches hallucinations, but 3-5× latency
- **No single approach wins everywhere — cost/latency/accuracy tradeoffs matter**

### §8 — Evaluation primer (optional, skip if short)
Brief mention of Ragas metrics (faithfulness, answer relevancy, context precision) and show a cached Ragas output. Don't run it live — it's slow.

### §9 — Takeaways
Markdown cell:
- 3 things learned
- Links to LangChain, CrewAI, Ragas docs
- Your contact info + lab

---

## 6. Ollama-on-Colab Gotchas (know these in advance)

- **Ollama takes 30-60s to start on first run.** Fine if you kick off §0 early.
- **The subprocess.Popen(['ollama', 'serve']) line must run in a background process.** Don't use `!ollama serve` — that blocks the cell forever.
- **Free Colab sometimes denies GPU.** llama3.2:3b runs on CPU in ~5-10s per response — acceptable but not snappy. If GPU is unavailable and latency bites, fall back to `llama3.2:1b`.
- **Colab sessions disconnect after 90 min idle.** Tell participants: if it disconnects, rerun §0 — it's fast.
- **Background Ollama process sometimes dies silently.** If a cell hangs, re-run §0.

**Backup plan: Groq API version**
If Ollama-on-Colab is unreliable on workshop day, the backup notebook uses Groq's free API with Llama 3 models. Identical code structure, just swap `ChatOllama` → `ChatGroq`. Faster responses, requires a free Groq API key. Have this ready as a second notebook tab.

---

## 7. Head-to-Head Comparison: Presenter Notes

This is your closer. Do this deliberately:

1. **Run the comparison cell while talking** — it takes 1-2 min per question × 5 questions × 3 systems ≈ 5-8 min total. Run it, then talk through the slides while it executes.
2. **Read the results OUT LOUD** one question at a time. Don't just display the table and move on.
3. **Call out the "Capital of Switzerland" row explicitly.** This is where RAG's grounding discipline shines — the plain LLM answers a question it shouldn't, the RAG systems refuse.
4. **Call out the "monster's name" row.** This is the emotional close — you promised a hallucination demo in slide 3, and here's the payoff.
5. **End with the cost/latency tradeoff:** show that agentic RAG took 5× longer. Not every problem needs agents.

---

## 8. Risks (Revised for Colab)

| Risk | Mitigation |
|---|---|
| Ollama install on Colab fails | Backup Groq-API notebook, ready to share link |
| Participant has no Google account | Super rare; pair them with a neighbor |
| Colab runtime disconnects mid-workshop | Cells are idempotent; rerun §0 → §4 takes 3 min |
| GPU unavailable on free tier | 1B model runs fine on CPU; keep model small |
| CrewAI version breaks | Pin exact versions in §0 install cell |
| Live code typo | Have "answer key" notebook in a second tab |

---

## 9. What Success Looks Like

After the workshop, attendees can:
- Explain what RAG is and why it reduces hallucination
- Open a Colab notebook, load a document, and run a basic RAG pipeline
- Articulate when Agentic RAG is worth the extra complexity
- Evaluate RAG outputs critically (not just accept them)
- Take the notebook home and adapt it to their own documents

---

## Appendix A: Answer Keys for Test Questions (Frankenstein)

For your own reference during the demo:
- **Monster's name:** No name. Called "the creature," "the wretch," "the fiend," "daemon," but never named.
- **Ship captain:** Robert Walton, writing letters to his sister Margaret.
- **Monster's reading:** *Paradise Lost* by Milton, *Plutarch's Lives*, *The Sorrows of Young Werter* by Goethe. He finds them in a leather portmanteau.
- **William's death:** The monster kills William (Victor's young brother) out of rage when he learns William is a Frankenstein.
- **Creation method:** Victor collects body parts from charnel houses, dissecting rooms, slaughterhouses. The animating method is deliberately vague — a "spark of being."

## Appendix B: Sanity-check Colab cell to put at the very top

```python
# Run this first. If any line fails, stop and ask for help.
import sys
assert sys.version_info >= (3, 10), "Need Python 3.10+"
!nvidia-smi || echo "No GPU — will run on CPU (slower but fine)"
print("Ready to proceed ✓")
```
