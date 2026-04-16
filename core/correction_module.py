"""
Correction Module
Triggered when Drift(t) > τ.

Three-Strike Escalation Strategy:
  Strike 1: Goal Reminder      — gentle nudge back to the goal
  Strike 2 (consecutive):      — Plan Regeneration (rewrite the plan from scratch)
  Strike 3+ (persistent drift) — Task Decomposition (break into smaller pieces)

Fix log:
  - Removed two orphaned unreachable lines at the bottom of the class body
    (`step_lower = step.lower()` and `return any(...)`) that were syntactically
    valid but never executed and referenced an undefined variable `step`.
"""

from typing import Dict, List
from prompts.prompt_templates import replan_prompt


class CorrectionModule:

    TASK_SIGNALS = [
        "search", "find", "read", "summarize", "categorize", "review",
        "paper", "source", "report", "research", "academic", "analyze",
        "extract", "synthesize", "gather", "collect", "identify",
    ]

    def __init__(self):
        self.consecutive_drift_count = 0

    def apply_correction(
        self,
        goal_data: dict,
        step:      str,
        context:   dict,
        llm=None,
    ) -> Dict:
        """
        Apply correction using the three-strike escalation strategy.

        Args:
            goal_data: Dict with 'goal_text' and 'subtasks' keys
            step:      The drifted observation/step text
            context:   Dict with 'steps' key (list of executed steps so far)
            llm:       Optional LLM for plan regeneration / decomposition

        Returns:
            Dict with corrected_step, strategy, strike, and optional plan lists.
        """
        print("\n⚠️  DRIFT DETECTED — APPLYING CORRECTION")

        goal_text  = goal_data["goal_text"]
        subtasks   = goal_data.get("subtasks", [])
        steps_done = [s for s in context.get("steps", []) if len(s) < 200]

        self.consecutive_drift_count += 1

        if self.consecutive_drift_count == 1:
            print("   ⚡ STRIKE 1: Goal Reminder")
            corrected = self._goal_reminder(goal_text, subtasks, steps_done)
            return {
                "corrected_step": corrected,
                "strategy":       "goal_reminder",
                "strike":         1,
            }

        if self.consecutive_drift_count == 2:
            print("   ⚡⚡ STRIKE 2: Plan Regeneration")
            remaining_plan = self._plan_regeneration(goal_text, subtasks, steps_done, llm)
            corrected = (
                remaining_plan[0]
                if remaining_plan
                else self._goal_reminder(goal_text, subtasks, steps_done)
            )
            return {
                "corrected_step": corrected,
                "strategy":       "plan_regeneration",
                "strike":         2,
                "remaining_plan": remaining_plan,
            }

        print("   ⚡⚡⚡ STRIKE 3+: Task Decomposition")
        substeps  = self._task_decomposition(goal_text, subtasks, steps_done, llm)
        corrected = (
            substeps[0]
            if substeps
            else self._goal_reminder(goal_text, subtasks, steps_done)
        )
        return {
            "corrected_step": corrected,
            "strategy":       "task_decomposition",
            "strike":         3,
            "substeps":       substeps,
        }

    def reset_strike_counter(self) -> None:
        """Reset consecutive drift counter when a successful step is achieved."""
        self.consecutive_drift_count = 0

    # ------------------------------------------------------------------ #
    #  Correction strategies                                               #
    # ------------------------------------------------------------------ #

    def _goal_reminder(self, goal_text: str, subtasks: list, steps_done: list) -> str:
        """
        Return the next expected subtask as a short concrete action (under 100 chars).
        Stored as the corrected step, so it must look like a real action sentence.
        """
        next_subtask = (
            subtasks[len(steps_done)] if len(steps_done) < len(subtasks)
            else subtasks[-1] if subtasks
            else "continue research"
        )

        action_map = {
            "find papers":        "Search academic databases for relevant papers",
            "search databases":   "Search Web of Science and Scopus for papers",
            "read abstracts":     "Read and extract key points from paper abstracts",
            "summarize":          "Summarize key findings from the collected papers",
            "extract findings":   "Extract and list the main research findings",
            "categorize":         "Group papers by theme and research area",
            "synthesize":         "Synthesize findings into a coherent summary",
            "generate report":    "Write a structured literature review report",
            "report":             "Write a final structured report",
            "gather information": "Collect academic sources on the topic",
            "process results":    "Analyze and organize research results",
            "understand task":    "Define the research scope and methodology",
        }
        return action_map.get(
            next_subtask.lower(),
            f"Continue: {next_subtask} for goal: {goal_text[:60]}",
        )

    def _plan_regeneration(
        self,
        goal_text:  str,
        subtasks:   list,
        steps_done: list,
        llm=None,
    ) -> List[str]:
        current_subtask = (
            subtasks[len(steps_done)] if len(steps_done) < len(subtasks)
            else subtasks[-1] if subtasks
            else "continue research"
        )

        if llm:
            prompt    = replan_prompt(goal_text, current_subtask)
            response  = llm.generate(prompt, max_length=100).strip()
            plan_steps = self._parse_plan_steps(response)
            if plan_steps:
                return plan_steps

        return [self._goal_reminder(goal_text, subtasks, steps_done)]

    def _task_decomposition(
        self,
        goal_text:  str,
        subtasks:   list,
        steps_done: list,
        llm=None,
    ) -> List[str]:
        """
        Strike 3: Break the current subtask into 3-5 smaller sequential micro-steps.
        """
        current_idx     = len(steps_done)
        current_subtask = (
            subtasks[current_idx] if current_idx < len(subtasks)
            else subtasks[-1] if subtasks
            else "continue research"
        )

        if llm:
            prompt = (
                f"Original goal: {goal_text}\n"
                f"Current task: {current_subtask}\n\n"
                f"Break this task into 3-5 smaller concrete steps that directly "
                f"support the original goal."
            )
            response  = llm.generate(prompt, max_length=120).strip()
            substeps  = self._parse_plan_steps(response)
            if len(substeps) >= 2:
                return substeps

        return self._decompose_subtask(current_subtask)[:4]

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_plan_steps(text: str) -> List[str]:
        lines = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            return lines
        if ";" in text:
            parts = [p.strip() for p in text.split(";") if p.strip()]
            if len(parts) > 1:
                return parts
        clean = text.strip()
        return [clean] if clean else []

    @staticmethod
    def _decompose_subtask(subtask: str) -> List[str]:
        """Map a subtask keyword to a list of granular micro-steps."""
        decomposition_map = {
            "find papers": [
                "Define search keywords from goal",
                "Search academic database 1",
                "Search academic database 2",
                "Compile unique results",
            ],
            "search": [
                "Define search parameters",
                "Execute search query",
                "Filter results by relevance",
                "Store results",
            ],
            "search databases": [
                "Query Web of Science",
                "Query Scopus",
                "Query Google Scholar",
                "Merge and deduplicate results",
            ],
            "read abstracts": [
                "Download paper metadata",
                "Extract abstract text",
                "Identify key sentences",
                "Compile findings",
            ],
            "extract findings": [
                "Identify main results in paper",
                "Extract methodology",
                "Note limitations",
                "List key citations",
            ],
            "summarize": [
                "Identify main themes",
                "Extract key quotes",
                "Synthesize points",
                "Create summary outline",
            ],
            "synthesize": [
                "Compare across papers",
                "Find common themes",
                "Identify differences",
                "Create integrated narrative",
            ],
            "categorize": [
                "Identify category dimensions",
                "Review each paper",
                "Assign to category",
                "Verify consistency",
            ],
            "group": [
                "Define grouping criteria",
                "Sort items by criteria",
                "Create groups",
                "Label each group",
            ],
            "generate report": [
                "Write introduction section",
                "Write methodology section",
                "Write findings section",
                "Write conclusion section",
                "Compile final report",
            ],
            "report": [
                "Create outline",
                "Fill each section",
                "Review for coherence",
                "Format final version",
            ],
            "gather information": [
                "Define information needs",
                "Identify sources",
                "Retrieve from sources",
                "Organize information",
            ],
            "collect": [
                "List items to collect",
                "Determine collection strategy",
                "Implement collection",
                "Verify completeness",
            ],
            "understand task": [
                "Read goal statement",
                "Identify key objectives",
                "Note constraints",
                "Create mental model",
            ],
            "process results": [
                "Parse raw results",
                "Validate data quality",
                "Apply filtering criteria",
                "Store processed results",
            ],
            "analyze": [
                "Define analysis framework",
                "Apply framework to data",
                "Extract insights",
                "Document findings",
            ],
        }

        subtask_lower = subtask.lower()
        for key, steps in decomposition_map.items():
            if key in subtask_lower:
                return steps

        # Default for unmatched subtasks
        return [
            f"Analyze '{subtask}' requirements",
            f"Plan approach for '{subtask}'",
            f"Execute '{subtask}' step-by-step",
            f"Verify completion of '{subtask}'",
        ]