# 🚀 AI App Builder

**A multi-agent AI system that converts natural language prompts into fully functional, multi-file web applications.**

Describe the app you want in plain English — a team of specialized AI agents plans, architects, and codes it for you, automatically.

Built with **LangGraph**, **Groq API**, and **Streamlit**.

---

## ✨ What It Does

Type a request like:

> "Build a personal portfolio website with an About Me section, a Projects section, and a Contact form."

...and the system automatically:
1. **Plans** the app's purpose and features
2. **Architects** the file structure and technical tasks
3. **Codes** each file and writes it to disk
4. Delivers a working, ready-to-run web application

---

## 🧠 How It Works — Multi-Agent Pipeline

```
User Prompt → Planner Agent → Architect Agent → Coder Agent → Working App
```

| Agent | Role |
|---|---|
| **Planner** | Understands the user's request and produces a structured, high-level app plan |
| **Architect** | Breaks the plan into a concrete file structure with detailed implementation tasks |
| **Coder** | Generates real, working code for each file and writes it to disk |

All agents are orchestrated using **LangGraph**, which manages shared state and controls the flow between agents.

---

## 🔒 Key Engineering Highlight: Fact-Locking Mechanism

LLMs can unintentionally alter or "hallucinate" specific user-provided details (like names, ages, or exact text) as they pass through multiple agent steps. To solve this, this project implements a **fact-locking mechanism**:

- User-provided facts are extracted into a structured, isolated dictionary at the start of the pipeline
- These facts are preserved untouched through the entire agent chain
- Final output is directly injected/verified against these locked facts — rather than relying on the LLM to "remember" them correctly across multiple regenerations

This significantly improves output reliability and accuracy for user-specific content.

---

## 🛠️ Tech Stack

- **Agent Orchestration:** LangGraph
- **LLM Provider:** Groq API (LLaMA / GPT-OSS models)
- **Frontend:** Streamlit
- **Language:** Python

---

## 📦 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/bibek373/ai-app-builder.git
cd ai-app-builder
```

### 2. Set up a virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free API key at [console.groq.com/keys](https://console.groq.com/keys)

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
ai-app-builder/
├── agents/
│   ├── planner.py       # Planner agent logic
│   ├── architect.py     # Architect agent logic
│   └── coder.py         # Coder agent logic
├── graph.py              # LangGraph pipeline orchestration
├── app.py                 # Streamlit frontend
├── state.py               # Shared state definition
├── generated_projects/    # Output folder for generated apps
└── requirements.txt
```

---

## 🎯 Example Use Cases

- Personal portfolio websites
- Landing pages
- Simple interactive tools (calculators, to-do lists)
- Informational/educational websites

---

## 📌 Current Limitations

- Generates static HTML/CSS/JavaScript applications only (no backend/database support yet)
- Best suited for single-page web applications

---

## 👤 Author

**Bibek Sah**
[GitHub](https://github.com/bibek373)

---

*Built as a hands-on exploration of multi-agent orchestration with LangGraph.*
