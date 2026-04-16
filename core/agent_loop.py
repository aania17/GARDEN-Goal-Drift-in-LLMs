"""
Layer 1 — Agent Loop
Orchestrates the 6-layer GARDEN architecture.

Improvement log:
  - Strike counter is now reset after every SUCCESSFUL (non-drifted,
    non-corrected) step. Previously it was never reset mid-task, so the
    escalation level leaked across subtasks — causing judge_only to issue
    14 corrections on only 2 actual drifts.
  - Drift detection now passes the current subtask text alongside the
    observation so the judge prompt has subtask context for a fairer rating.
  - _seed_rag() now calls executor.reset() after seeding so accumulated seed
    papers don't inflate the paper count for the actual task run.
"""

import json
from typing import Dict, List


class AgentLoop:

    def __init__(self):
        try:
            from core.goal_decomposer import GoalDecomposer
            from core.reasoning_engine import ReasoningEngine
            from core.executor import Executor
            from core.rag_module import RAGModule
            from core.drift_detector import DriftDetector
            from core.correction_module import CorrectionModule
            from core.evaluation_layer import EvaluationLayer

            self.goal_decomposer    = GoalDecomposer()
            self.executor           = Executor()
            self.rag_module         = RAGModule()
            self.drift_detector     = DriftDetector()
            self.correction_module  = CorrectionModule()
            self.evaluation_layer   = EvaluationLayer()
            self.context_memory     = {}

            self.llm = None
            try:
                from utils.llm_engine import LLMEngine
                self.llm = LLMEngine()
            except Exception as e:
                print(f"Note: LLM engine not available: {e}. Using heuristics.")

            self.reasoning_engine = ReasoningEngine(llm=self.llm)

        except ImportError as e:
            print(f"Warning: Some modules not available: {e}")

    def run(self, goal_text: str, agent_mode: str = "hybrid") -> Dict:
        """
        Execute the GARDEN pipeline.

        Args:
            goal_text:   The user's research goal
            agent_mode:  One of 'baseline', 'embedding_only', 'judge_only', 'hybrid'
        """
        result = {
            "goal":                      goal_text,
            "agent_mode":                agent_mode,
            "steps":                     [],
            "observations":              [],
            "drift_detected":            False,
            "corrections_applied":       [],
            "final_answer":              None,
            "total_corrections_applied": 0,
            "final_evaluation":          None,
        }

        try:
            self.evaluation_layer.set_goal_text(goal_text)
            self.evaluation_layer.set_agent_mode(agent_mode)

            self.drift_detector.reset()
            self.correction_module.reset_strike_counter()
            self.executor.reset()
            self.context_memory = {}

            persistent_alignment_context  = (
                "You must remain aligned to the original goal. "
                "Do not drift to unrelated topics."
            )
            latest_correction_instruction = ""
            execution_context             = ""
            executed_steps                = []

            max_corrections_by_mode = {
                "baseline":       0,
                "embedding_only": 1,
                "judge_only":     2,
                "hybrid":         3,
            }
            allowed_corrections = max_corrections_by_mode.get(agent_mode, 0)

            # Layer 1: Goal Decomposition
            subtasks = self.goal_decomposer.decompose(goal_text)
            result["steps"] = list(subtasks)

            # Layer 2: Seed RAG with real papers
            self._seed_rag(goal_text)

            # Layers 3-6: Main reasoning loop
            idx = 0
            while idx < len(subtasks):
                current_subtask = subtasks[idx]

                executor_prompt = self.executor.build_executor_prompt(
                    goal_text,
                    persistent_alignment_context,
                    execution_context,
                    current_subtask,
                    latest_correction_instruction,
                )

                action_plan = self.reasoning_engine.reason(
                    executor_prompt,
                    goal_text,
                    current_subtask,
                    execution_context,
                    persistent_alignment_context,
                    latest_correction_instruction,
                )

                execution_result = self.executor.execute(action_plan, goal_text)
                observation_text = (
                    execution_result.get("observation", str(execution_result))
                    if isinstance(execution_result, dict)
                    else str(execution_result)
                )

                execution_context += (
                    f"SUBTASK: {current_subtask}\n"
                    f"ACTION: {action_plan}\n"
                    f"OBSERVATION: {observation_text}\n"
                )
                executed_steps.append(current_subtask)

                result["observations"].append({
                    "subtask":          current_subtask,
                    "action":           action_plan,
                    "observation":      observation_text,
                    "execution_result": execution_result,
                })

                # Drift detection — pass subtask for richer judge context
                drift_result = self.drift_detector.detect(
                    observation=observation_text,
                    goal_text=goal_text,
                    mode=agent_mode,
                    llm=self.llm,
                    subtask=current_subtask,   # NEW: subtask context for judge
                )

                correction_result = None
                corrected         = False

                if drift_result.get("drift_detected", False):
                    result["drift_detected"] = True
                    if (agent_mode != "baseline"
                            and result["total_corrections_applied"] < allowed_corrections):
                        correction_result = self.correction_module.apply_correction(
                            goal_data={"goal_text": goal_text, "subtasks": subtasks},
                            step=observation_text,
                            context={"steps": executed_steps},
                            llm=self.llm,
                        )
                        result["corrections_applied"].append(correction_result)
                        result["total_corrections_applied"] += 1
                        corrected = True

                        latest_correction_instruction = correction_result.get(
                            "corrected_step", ""
                        )
                        persistent_alignment_context += (
                            f"{correction_result.get('strategy', 'correction').replace('_', ' ').title()}: "
                            f"{latest_correction_instruction}\n"
                        )

                        if correction_result.get("remaining_plan"):
                            subtasks = correction_result["remaining_plan"]
                            idx = -1
                        elif correction_result.get("substeps"):
                            subtasks = correction_result["substeps"] + subtasks[idx + 1:]
                            idx = -1
                else:
                    # ── KEY FIX: reset strike counter on every successful step ──
                    # Without this, the escalation level leaks across subtasks,
                    # causing judge_only to reach Strike 2/3 on later subtasks
                    # even when the agent hasn't drifted consecutively.
                    self.correction_module.reset_strike_counter()

                self.evaluation_layer.record_step(
                    drift_result,
                    corrected=corrected,
                    correction_info=correction_result,
                )

                idx += 1

            final_observations  = [o.get("observation", "") for o in result["observations"]]
            result["final_answer"] = (
                " ".join(final_observations) if final_observations else None
            )

            self.evaluation_layer.mark_task_complete()
            result["final_evaluation"] = self.evaluation_layer.to_dict()
            self.context_memory = {
                "persistent_alignment_context": persistent_alignment_context,
                "execution_context":            execution_context,
                "executed_steps":               executed_steps,
            }

        except Exception as e:
            result["error"] = str(e)
            print(f"Error in agent loop: {e}")
            import traceback
            traceback.print_exc()

        return result

    def _seed_rag(self, goal_text: str) -> None:
        """
        Seed the RAG module with real arXiv papers for this goal.
        Resets executor paper store after seeding so seed papers don't inflate
        the paper count during the actual task run.
        """
        try:
            topics = self._extract_topics_from_goal(goal_text)
            for topic in topics:
                try:
                    papers_result = self.executor.execute(
                        f"Search for papers on {topic}", goal_text
                    )
                    for paper in papers_result.get("papers", []):
                        self.rag_module.add_document({
                            "id":      paper.get("arxiv_id", paper.get("title", "")),
                            "title":   paper.get("title", ""),
                            "content": paper.get("summary", ""),
                            "source":  "arxiv",
                            "topic":   topic,
                        })
                except Exception as e:
                    print(f"Warning: RAG seeding failed for '{topic}': {e}")

            # Reset executor so seed papers don't bleed into task execution
            self.executor.reset()

        except Exception as e:
            print(f"Warning: RAG seeding failed: {e}")

    def _extract_topics_from_goal(self, goal: str) -> List[str]:
        cleaned = goal.lower()
        for prefix in [
            "conduct a systematic literature review on ",
            "conduct a literature review on ",
            "research ", "investigate ", "analyze ", "study ", "review ",
        ]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break

        topics = [cleaned[:100]]

        keywords = [
            "machine learning", "deep learning", "artificial intelligence",
            "neural network", "healthcare", "medical", "cybersecurity",
            "quantum", "blockchain", "renewable energy", "climate",
            "natural language", "protein", "autonomous",
        ]
        for kw in keywords:
            if kw in cleaned:
                topics.append(kw)

        return list(set(topics))[:3]

    def compare_modes(self, goal_text: str) -> Dict:
        """Run all four modes on a single goal for comparison."""
        modes      = ["baseline", "embedding_only", "judge_only", "hybrid"]
        results    = {mode: self.run(goal_text, agent_mode=mode) for mode in modes}
        analysis   = {
            "goal":    goal_text,
            "results": results,
            "analysis": {
                "drift_detection_effectiveness": {
                    m: results[m].get("drift_detected", False) for m in modes
                },
                "correction_efficiency": {
                    m: len(results[m].get("corrections_applied", [])) for m in modes
                },
                "evaluation_scores": {
                    m: (results[m].get("final_evaluation", {}) or {})
                       .get("metrics", {}).get("success", 0)
                    for m in modes
                },
            },
        }
        return analysis

    def get_context_memory(self) -> Dict:
        return self.context_memory

    def clear_context_memory(self) -> None:
        self.context_memory = {}