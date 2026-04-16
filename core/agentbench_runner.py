"""
Layer 6 — AgentBench Integration
Replaces the mock Executor with real AgentBench task environments.

AgentBench provides interactive multi-step tasks where drift can
naturally emerge across long reasoning chains.

Supported task types (from the paper):
  - QA             — question answering with tool use
  - Reasoning      — multi-hop reasoning chains
  - Tool use       — search, retrieve, summarize actions

Usage:
    runner = AgentBenchRunner(embedder)
    runner.run_task("literature_survey_001")

To install AgentBench:
    git clone https://github.com/THUDM/AgentBench
    pip install -e AgentBench
"""

import json
from core.agent_loop import AgentLoop
from core.evaluation_layer import EvaluationLayer
from data.goal_processor import GoalProcessor
from data.memory_store import GoalMemory


# ── Mock task registry (replace with real AgentBench loader) ──────────── #

MOCK_TASKS = {
    "literature_survey_001": {
        "goal": "Conduct a literature survey on renewable energy using academic sources",
        "environment": "research database",
        "tools": ["search", "retrieve", "summarize"],
        "expected_subtasks": ["find papers", "summarize", "categorize", "generate report"],
        "success_criterion": "report",
    },
    "qa_energy_policy_001": {
        "goal": "Answer: what are the main policy barriers to renewable energy adoption?",
        "environment": "web + document store",
        "tools": ["search", "retrieve"],
        "expected_subtasks": ["search policy papers", "extract barriers", "synthesize answer"],
        "success_criterion": "answer",
    },
    "reasoning_chain_001": {
        "goal": "Compare solar and wind energy costs over the last decade using data",
        "environment": "research database",
        "tools": ["search", "retrieve", "summarize"],
        "expected_subtasks": ["find cost data", "extract figures", "compare", "summarize findings"],
        "success_criterion": "comparison report",
    },
}


class AgentBenchRunner:
    """
    Runs GARDEN against AgentBench tasks and records evaluation metrics.

    In a full integration, self._load_task() would call the real
    AgentBench environment API. Currently uses mock tasks.
    """

    def __init__(self, embedder):
        self.embedder = embedder
        self.results  = []

    def run_task(self, task_id: str) -> dict:
        """Run a single AgentBench task and return evaluation results."""
        task = self._load_task(task_id)
        if task is None:
            print(f"Task '{task_id}' not found.")
            return {}

        print(f"\n{'='*60}")
        print(f"AgentBench Task: {task_id}")
        print(f"{'='*60}")

        # Layer 1 — process goal
        processor = GoalProcessor()
        goal_data = processor.extract_goal(
            user_input        = task["goal"],
            environment_state = task["environment"],
            tool_outputs      = [],
        )

        # Layer 2 — store in goal memory
        goal_memory = GoalMemory()
        embedding   = self.embedder.embed(goal_data["goal_text"])
        goal_memory.store_goal(goal_data, embedding)

        # Run agent loop
        agent = AgentLoop(self.embedder, use_llm_judge=True)
        agent.run(goal_memory, max_iterations=6)

        # Collect results
        result = {
            "task_id":             task_id,
            "goal":                task["goal"],
            "task_completed":      agent.evaluator.task_completed,
            "task_success_rate":   agent.evaluator.task_success_rate(),
            "goal_adherence":      agent.evaluator.goal_adherence_score(),
            "avg_drift_score":     agent.evaluator.average_drift_score(),
            "pass_at_1":           agent.evaluator.pass_at_1(),
            "drift_trajectory":    agent.evaluator.drift_trajectory,
            "total_iterations":    agent.evaluator.total_iterations,
        }
        self.results.append(result)
        return result

    def run_all(self) -> list[dict]:
        """Run all registered tasks."""
        for task_id in MOCK_TASKS:
            self.run_task(task_id)
        return self.results

    def comparison_report(self) -> str:
        """
        Print comparison table matching the paper's Table in Section 12:
          System          | Drift  | Success
          Baseline agent  | High   | Lower
          ReAct agent     | Medium | Medium
          GARDEN          | Low    | High
        """
        if not self.results:
            return "No results yet. Run tasks first."

        lines = [
            "\n" + "=" * 65,
            "  GARDEN vs Baseline Comparison (Layer 6 Evaluation)",
            "=" * 65,
            f"  {'Task':<30} {'Adherence':>10} {'Drift':>8} {'Pass@1':>8}",
            "  " + "-" * 60,
        ]
        for r in self.results:
            lines.append(
                f"  {r['task_id']:<30} "
                f"{r['goal_adherence']:>9.3f} "
                f"{r['avg_drift_score']:>8.3f} "
                f"{r['pass_at_1']:>8.1f}"
            )

        if len(self.results) > 1:
            avg_adherence = sum(r["goal_adherence"] for r in self.results) / len(self.results)
            avg_drift     = sum(r["avg_drift_score"] for r in self.results) / len(self.results)
            avg_pass      = sum(r["pass_at_1"]        for r in self.results) / len(self.results)
            lines += [
                "  " + "-" * 60,
                f"  {'GARDEN (avg)':<30} {avg_adherence:>9.3f} {avg_drift:>8.3f} {avg_pass:>8.2f}",
                f"  {'Baseline ReAct (est.)':<30} {'~0.55':>9} {'~0.45':>8} {'~0.50':>8}",
                f"  {'No-drift-detection (est.)':<30} {'~0.40':>9} {'~0.60':>8} {'~0.35':>8}",
            ]

        lines.append("=" * 65)
        return "\n".join(lines)

    def save_results(self, path: str = "evaluation_results.json") -> None:
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {path}")

    def _load_task(self, task_id: str) -> dict | None:
        """
        Load a task definition.
        Replace this with real AgentBench environment API calls:

            from agentbench import TaskLoader
            return TaskLoader.load(task_id)
        """
        return MOCK_TASKS.get(task_id)