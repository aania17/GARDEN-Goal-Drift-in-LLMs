
# GARDEN
## Goal-Anchored Retrieval-Driven Drift Evaluation Network

A research implementation of a long-horizon LLM agent designed to actively detect and mitigate **context drift**—the phenomenon where an AI agent gradually stops following its original goal as intermediate reasoning steps accumulate.

---

## 🛑 The Problem: Context Drift

When LLM agents run over extended horizons, they increasingly condition their subsequent actions on recent context rather than their primary system instruction. This deterioration of task focus leads to:
* **Going Off-Topic:** The agent abandons the core task to pursue tangential subtopics.
* **Objective Forgetting:** The original user intent is lost deep in the context window.
* **Detail Fixation:** The agent loops endlessly on a minor detail or failed subtask.

GARDEN solves this through three integrated mechanisms: persistent goal anchoring, continuous dual-metric drift monitoring, and adaptive corrective planning.

---

## 🧠 Architecture Overview

The system is organized into a modular 6-layer architecture, designed to act as a robust evaluation harness for multi-step reasoning.

```text
Layer 1 — Input Layer
  User Goal + Environment State
        │
        ▼
Layer 2 — Goal Memory Layer
  Persistent: Goal Text + Goal Embedding Vector (G)
        │
        ▼
Layer 3 — Prompt Engineering Layer
  Goal Anchoring Prompt | Reflection Template | Structured Reasoning Template
        │
        ▼
Layer 4 — Agent Workflow Layer  ◄─────────────────────────────┐
  Goal Decomposer → Context Memory → Executor                 │
       ↑                                                      │
       │                                                      │
Layer 4.5 — Drift Detection & Correction Module (Core Innovation)
  Metric 1: Embedding Cosine Sim  |  Metric 2: LLM Semantic Judge
  Strike 1: Nudge  |  Strike 2: Replan  |  Strike 3: Decompose
       │
       ▼
Layer 5 — RAG Module
  FAISS Vector Store (Domain Knowledge & Execution Traces)
       │
       ▼
Layer 6 — Evaluation Layer
  Strict Trajectory Metrics: Task Success, Goal Adherence, Pass@1
```

### The Core Innovation: Layer 4.5
Instead of waiting for an agent to fail completely, GARDEN evaluates every single reasoning step before it is committed to memory. If the step deviates from the original goal (Drift > $\tau$), a **Three-Strike Escalation Strategy** intervenes:
1. **Strike 1 (Contextual Nudge):** A subtle goal reminder.
2. **Strike 2 (Plan Regeneration):** The agent is forced to rewrite its action plan from scratch.
3. **Strike 3 (Task Decomposition):** The stalled subtask is broken down into smaller, sequential micro-steps.

---

## 📊 Ablation Study Results

To validate the architecture, we conducted an ablation study across 10 synthetic multi-hop academic research tasks. We isolated the drift detection mechanisms to determine the most effective way to monitor agent alignment.

**Success Criteria:** To achieve a "Success," the agent had to complete the task, maintain an average trajectory alignment $\ge 0.65$, and suffer $\le 2$ uncorrected drifts.

| Mode | Task Success Rate | Avg Alignment (0-1) | Uncorrected Drifts | Interventions/Task |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Standard ReAct)** | 40% | 0.744 | 2.3 | 0.0 |
| **Embedding-Only (Math)** | 70% | 0.694 | 1.4 | 1.0 |
| **Hybrid (Staged)** | 70% | 0.688 | 0.1 | 0.8 |
| **Judge-Only (Semantic)** | **80%** | **0.700** | **0.0** | **0.7** |

### Key Findings
1. **Context Drift is Fatal without Monitoring:** The Baseline agent failed 60% of tasks due to uncorrected goal drift (averaging 2.3 drifts per run), proving that standard ReAct loops are highly unstable over long horizons.
2. **Semantic Understanding > Lexical Similarity:** The **Judge-Only** mode was the absolute peak performer (80% success). It was surgically precise, driving uncorrected drift down to an absolute zero while requiring the fewest interventions (0.7).
3. **The "Noise" of Embeddings:** While Cosine Similarity (Embedding-Only) provides a decent safety net, it suffers from lexical rigidity. As the agent's vocabulary naturally shifts from "searching" to "synthesizing", the embedding incorrectly flags it as a drift event. This false-positive noise is why the Hybrid mode underperformed the pure Judge mode.

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10+
* **Ollama:** GARDEN is built to run entirely locally using `llama3.2`.
  * Install from [ollama.com](https://ollama.com)
  * Run: `ollama pull llama3.2`

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/yourusername/GARDEN.git](https://github.com/yourusername/GARDEN.git)
cd GARDEN
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 3. Verify Setup
Run the interactive quickstart guide to verify your dependencies and project structure:
```bash
python QUICKSTART.py
```

---

## 🔬 Running the Ablation Study

To reproduce the research findings and generate the JSON metrics:
```bash
python main.py --mode ablation
```
*This runs 40 total executions (10 tasks × 4 modes) and takes approximately 20-30 minutes depending on your hardware.*

### Generate Visualizations
Once the study completes, generate the publication-ready PNG figures:
```bash
python visualize_ablation.py
```
Outputs will be saved in the `visualizations/` directory, including:
1. `01_task_success_rate.png`
2. `02_goal_adherence.png`
3. `03_correction_efficiency.png`
4. `04_run_success_rate.png`
5. `05_performance_heatmap.png`

### Interactive Testing
To test a single task or compare modes interactively:
```bash
# Run a single mode
python main.py --mode single --agent-mode judge_only

# Interactively pick a task and compare all modes
python main.py --mode compare
```

---

## 🛠️ Troubleshooting

* **`⚠️ Model 'llama3.2' not found`** — Run `ollama pull llama3.2` and wait for the download to complete.
* **`ModuleNotFoundError`** — Ensure your virtual environment is activated and `requirements.txt` is installed.
* **Ollama Connection Refused** — Ensure the Ollama app is running in the background (check your system tray) or run `ollama serve` in a separate terminal.
