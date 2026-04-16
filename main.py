"""
GARDEN: Goal-Aware Research Data Evaluation Network
Main entry point for ablation study automation and real-world data integration.

This script orchestrates the complete ablation study:
  - 10 synthetic research tasks
  - 4 agent modes (baseline, embedding_only, judge_only, hybrid)
  - Automated metric collection and JSON export

Fix log:
  - _compute_summary_statistics() now reads 'success' (compute_success) from
    metrics dict rather than 'task_success_rate' which was always 1.0 and made
    the headline chart uninformative.
  - _print_summary() prints the corrected success_rate alongside avg_alignment
    and raw drift/correction counts for a clearer ablation picture.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List
from core.agent_loop import AgentLoop
from core.evaluation_layer import EvaluationLayer


# ============================================================================
# SYNTHETIC DRIFT TASKS
# ============================================================================

SYNTHETIC_DRIFT_TASKS = [
    {
        "id": 1,
        "goal": "Conduct a systematic literature review on machine learning applications in medical diagnosis, focusing on CNNs for image classification. Start by finding at least 5 papers about neural networks in healthcare, then summarize their methodologies, and finally categorize them by disease type.",
        "subtasks": ["Find papers on ML in medical diagnosis", "Summarize methodologies", "Categorize by disease type", "Generate final report"],
    },
    {
        "id": 2,
        "goal": "Research the evolution of natural language processing from rule-based systems to transformer models. First, identify key papers on NLP history, then extract the main innovations in each era, synthesize the progression, and create a timeline.",
        "subtasks": ["Find NLP history papers", "Extract key innovations", "Synthesize progression", "Create timeline"],
    },
    {
        "id": 3,
        "goal": "Investigate climate change impacts on coastal ecosystems. Search for scientific papers on rising sea levels and marine biodiversity, summarize the findings, categorize impacts by region, and generate policy recommendations.",
        "subtasks": ["Find papers on coastal ecosystems", "Summarize findings", "Categorize impacts by region", "Generate recommendations"],
    },
    {
        "id": 4,
        "goal": "Analyze quantum computing breakthroughs and applications. Locate papers on quantum algorithms, extract technical details, categorize by application domain (optimization, simulation, cryptography), and assess practical feasibility.",
        "subtasks": ["Find quantum computing papers", "Extract technical details", "Categorize by application", "Assess feasibility"],
    },
    {
        "id": 5,
        "goal": "Review recent advances in renewable energy technologies. Search for papers on solar, wind, and energy storage solutions, synthesize the technological progress, compare efficiency improvements, and evaluate cost trends.",
        "subtasks": ["Find renewable energy papers", "Synthesize progress", "Compare efficiencies", "Evaluate costs"],
    },
    {
        "id": 6,
        "goal": "Study the role of artificial intelligence in cybersecurity threat detection. Find papers on AI/ML security applications, extract attack detection methods, categorize by threat type, and summarize best practices.",
        "subtasks": ["Find cybersecurity papers", "Extract detection methods", "Categorize by threat type", "Summarize practices"],
    },
    {
        "id": 7,
        "goal": "Examine ethical considerations in autonomous vehicle development. Research papers on AI ethics, autonomous driving regulations, and safety concerns. Synthesize different ethical frameworks and compile policy recommendations.",
        "subtasks": ["Find ethics papers", "Extract frameworks", "Review regulations", "Compile recommendations"],
    },
    {
        "id": 8,
        "goal": "Investigate protein folding prediction using deep learning. Locate papers on AlphaFold and related methods, extract key algorithmic components, analyze computational complexity, and evaluate biological impact.",
        "subtasks": ["Find protein folding papers", "Extract algorithms", "Analyze complexity", "Evaluate impact"],
    },
    {
        "id": 9,
        "goal": "Research the impact of social media on mental health. Find peer-reviewed studies on social media usage, depression, and anxiety, categorize findings by age group, synthesize risk factors, and identify protective strategies.",
        "subtasks": ["Find mental health papers", "Categorize by age group", "Synthesize risk factors", "Identify strategies"],
    },
    {
        "id": 10,
        "goal": "Review blockchain applications beyond cryptocurrency. Search for papers on smart contracts, supply chain tracking, and healthcare records. Categorize applications by industry and assess scalability challenges.",
        "subtasks": ["Find blockchain papers", "Categorize by industry", "Extract use cases", "Assess challenges"],
    },
]


# ============================================================================
# ABLATION STUDY ORCHESTRATION
# ============================================================================

class AblationStudyRunner:
    """Orchestrates the complete ablation study across all tasks and modes."""

    def __init__(self, output_dir: str = "ablation_results"):
        self.output_dir = output_dir
        self.modes      = ["baseline", "embedding_only", "judge_only", "hybrid"]
        self.all_results = []

        os.makedirs(output_dir, exist_ok=True)

    def run_complete_study(self) -> Dict:
        """
        Run the complete ablation study:
          - Execute all 10 tasks through all 4 modes
          - Collect metrics for each run
          - Export results to JSON

        Returns:
            Dictionary with comprehensive study results.
        """
        print("\n" + "=" * 80)
        print("GARDEN ABLATION STUDY: Complete Execution")
        print("=" * 80)
        print(f"📊 Tasks: {len(SYNTHETIC_DRIFT_TASKS)}")
        print(f"🔄 Modes: {len(self.modes)}")
        print(f"📈 Total runs: {len(SYNTHETIC_DRIFT_TASKS) * len(self.modes)}")
        print("=" * 80)

        study_start = datetime.now()
        task_results = []

        for task_idx, task in enumerate(SYNTHETIC_DRIFT_TASKS, 1):
            print(f"\n{'─' * 80}")
            print(f"📝 TASK {task_idx}/{len(SYNTHETIC_DRIFT_TASKS)}: {task['goal'][:60]}...")
            print(f"{'─' * 80}")

            task_data = {
                "task_id":     task["id"],
                "goal":        task["goal"],
                "subtasks":    task["subtasks"],
                "mode_results": {},
            }

            for mode in self.modes:
                print(f"\n  ▶ Running {mode.upper():15} mode...", end=" ", flush=True)
                try:
                    agent  = AgentLoop()
                    result = agent.run(task["goal"], agent_mode=mode)

                    evaluation         = result.get("final_evaluation", {})
                    evaluation_metrics = (
                        evaluation.get("metrics", {})
                        if isinstance(evaluation, dict) else {}
                    )

                    task_data["mode_results"][mode] = {
                        "mode":              mode,
                        "execution_success": result.get("error") is None,
                        "metrics": {
                            # Use compute_success (quality-gated) not task_success_rate
                            "task_success_rate":   evaluation_metrics.get("task_success_rate", 0.0),
                            "compute_success":     evaluation_metrics.get("success", 0),
                            "goal_adherence_score":evaluation_metrics.get("goal_adherence_score", 0.0),
                            "average_alignment":   evaluation_metrics.get("average_alignment", 0.0),
                            "drift_count":         evaluation_metrics.get("drift_count", 0),
                            "correction_count":    evaluation_metrics.get("correction_count", 0),
                            "drift_detected":      result.get("drift_detected", False),
                            "corrections_applied": len(result.get("corrections_applied", [])),
                            "observations_count":  len(result.get("observations", [])),
                        },
                        "full_result": result,
                    }
                    print("✓ Complete")

                except Exception as e:
                    print(f"✗ Failed: {str(e)}")
                    task_data["mode_results"][mode] = {
                        "mode":              mode,
                        "execution_success": False,
                        "error":             str(e),
                        "metrics":           {},
                    }

            task_results.append(task_data)

        study_end = datetime.now()
        study_results = {
            "study_metadata": {
                "start_time":             study_start.isoformat(),
                "end_time":               study_end.isoformat(),
                "total_duration_seconds": (study_end - study_start).total_seconds(),
                "total_tasks":            len(SYNTHETIC_DRIFT_TASKS),
                "total_modes":            len(self.modes),
                "total_runs":             len(SYNTHETIC_DRIFT_TASKS) * len(self.modes),
            },
            "modes":              self.modes,
            "task_results":       task_results,
            "summary_statistics": self._compute_summary_statistics(task_results),
        }

        self._export_results(study_results)
        return study_results

    def _compute_summary_statistics(self, task_results: List[Dict]) -> Dict:
        """
        Aggregate per-mode statistics across all tasks.

        Uses 'success' (compute_success) as the primary success indicator
        rather than 'task_success_rate' which only checks loop completion.
        """
        summary     = {"by_mode": {}}
        total_tasks = len(task_results)

        for mode in self.modes:
            mode_metrics = {
                "successful_tasks":  0,
                "alignment_scores":  [],
                "total_drifts":      0,
                "total_corrections": 0,
            }

            for task_data in task_results:
                mode_result = task_data["mode_results"].get(mode, {})
                if not mode_result.get("execution_success", False):
                    continue

                metrics = mode_result.get("metrics", {})

                # Primary success criterion: compute_success (quality-gated)
                if metrics.get("compute_success", 0) == 1:
                    mode_metrics["successful_tasks"] += 1

                avg_alignment = metrics.get("average_alignment", 0.0)
                mode_metrics["alignment_scores"].append(avg_alignment)

                mode_metrics["total_drifts"]      += metrics.get("drift_count", 0)
                mode_metrics["total_corrections"]  += metrics.get("correction_count", 0)

            mode_metrics["success_rate"] = (
                mode_metrics["successful_tasks"] / total_tasks
                if total_tasks > 0 else 0.0
            )
            mode_metrics["avg_alignment"] = (
                sum(mode_metrics["alignment_scores"]) / len(mode_metrics["alignment_scores"])
                if mode_metrics["alignment_scores"] else 0.0
            )
            mode_metrics["avg_drift"] = (
                mode_metrics["total_drifts"] / total_tasks
                if total_tasks > 0 else 0.0
            )
            mode_metrics["avg_corrections"] = (
                mode_metrics["total_corrections"] / total_tasks
                if total_tasks > 0 else 0.0
            )

            summary["by_mode"][mode] = mode_metrics

        return summary

    def _export_results(self, study_results: Dict) -> None:
        full_path = os.path.join(self.output_dir, "ablation_study_full.json")
        with open(full_path, "w") as f:
            json.dump(study_results, f, indent=2, default=str)
        print(f"\n✅ Full results saved to: {full_path}")

        summary_path = os.path.join(self.output_dir, "ablation_study_summary.json")
        summary_only = {
            "study_metadata":     study_results["study_metadata"],
            "modes":              study_results["modes"],
            "summary_statistics": study_results["summary_statistics"],
        }
        with open(summary_path, "w") as f:
            json.dump(summary_only, f, indent=2, default=str)
        print(f"✅ Summary saved to: {summary_path}")

        self._print_summary(study_results["summary_statistics"])

    def _print_summary(self, summary: Dict) -> None:
        """Print per-mode statistics in a clean research table format."""
        print("\n" + "=" * 80)
        print("ABLATION STUDY SUMMARY")
        print(f"{'Mode':<18} {'Success':>8} {'Avg Align':>10} {'Avg Drifts':>12} {'Avg Corrects':>14}")
        print("=" * 80)

        for mode in self.modes:
            if mode not in summary["by_mode"]:
                continue
            s = summary["by_mode"][mode]
            print(
                f"{mode:<18} "
                f"{s['success_rate']:>7.0%} "
                f"{s['avg_alignment']:>10.3f} "
                f"{s['avg_drift']:>12.1f} "
                f"{s['avg_corrections']:>14.1f}"
            )
        print("=" * 80)


# ============================================================================
# CLI
# ============================================================================

def run_single_goal(goal_text: str, agent_mode: str = "hybrid") -> Dict:
    agent = AgentLoop()
    return agent.run(goal_text, agent_mode)


def main():
    parser = argparse.ArgumentParser(
        description="GARDEN Ablation Study: Automated comparative analysis"
    )
    parser.add_argument(
        "--mode", type=str, default="ablation",
        choices=["single", "ablation", "compare"],
        help="single=one task, ablation=full study, compare=task comparison",
    )
    parser.add_argument(
        "--goal", type=str, default=None,
        help="Custom goal for single mode",
    )
    parser.add_argument(
        "--agent-mode", type=str, default="hybrid",
        choices=["baseline", "embedding_only", "judge_only", "hybrid"],
        help="Agent mode for single execution",
    )
    parser.add_argument(
        "--output-dir", type=str, default="ablation_results",
        help="Directory for exporting results",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("GARDEN: Goal-Aware Research Data Evaluation Network")
    print("=" * 80)

    if args.mode == "ablation":
        runner = AblationStudyRunner(output_dir=args.output_dir)
        runner.run_complete_study()
        print("\n✅ Ablation study complete!")

    elif args.mode == "single":
        goal = args.goal or SYNTHETIC_DRIFT_TASKS[0]["goal"]
        print(f"\n🎯 Goal: {goal[:70]}...")
        print(f"📊 Mode: {args.agent_mode}")
        print("\n▶ Executing agent loop...")
        result = run_single_goal(goal, args.agent_mode)
        print("\n" + "=" * 80)
        print("EXECUTION RESULTS")
        print("=" * 80)
        print(json.dumps(result, indent=2, default=str))

    else:  # compare
        print("\n🔬 Comparative Analysis Mode")
        for i, task in enumerate(SYNTHETIC_DRIFT_TASKS, 1):
            print(f"  {i}. {task['goal'][:60]}...")
        task_num = input("\nEnter task number (1-10): ").strip()
        try:
            task_num = int(task_num)
            if 1 <= task_num <= 10:
                task = SYNTHETIC_DRIFT_TASKS[task_num - 1]
                print(f"\n▶ Running modes on: {task['goal'][:60]}...")
                comparison = {}
                for mode in ["baseline", "embedding_only", "judge_only", "hybrid"]:
                    print(f"\n  {mode}...", end=" ", flush=True)
                    comparison[mode] = run_single_goal(task["goal"], mode)
                    print("✓")
                comparison_path = os.path.join(
                    args.output_dir, f"task_{task_num}_comparison.json"
                )
                os.makedirs(args.output_dir, exist_ok=True)
                with open(comparison_path, "w") as f:
                    json.dump(comparison, f, indent=2, default=str)
                print(f"\n✅ Comparison saved to: {comparison_path}")
            else:
                print("Invalid task number.")
        except ValueError:
            print("Invalid input.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()