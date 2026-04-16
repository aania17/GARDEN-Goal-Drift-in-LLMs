"""
Layer 1 — Input Layer
Processes all four inputs shown in the diagram:
  - User Goal
  - Task Context
  - Environment State
  - Tool Outputs
"""


class GoalProcessor:
    """Extracts the core goal representation from the raw user instruction."""

    # Keywords that signal specific task constraints
    CONSTRAINT_MAP = {
        "academic":    "academic sources only",
        "peer-reviewed": "peer-reviewed sources",
        "summary":     "structured summary required",
        "report":      "report generation required",
        "peer":        "peer-reviewed sources",
        "recent":      "recent publications preferred",
        "categorize":  "categorization of findings required",
        "evidence":    "evidence-based conclusions required",
        "structured":  "structured output required",
        "step-by-step": "sequential structured output required",
        "compare":     "direct comparison required",
        "plan":        "actionable plan required",
    }

    # Subtask templates keyed on goal type signals
    SUBTASK_TEMPLATES = {
        "survey":   ["find papers", "summarize", "categorize", "generate report"],
        "research": ["search databases", "read abstracts", "extract findings", "synthesize"],
        "compare":  ["define comparison criteria", "gather data on each option", "compare side by side", "synthesize conclusions"],
        "plan":     ["define scope and objectives", "identify stakeholders and sources", "draft the plan", "review and finalize"],
        "design":   ["define scope and objectives", "identify stakeholders and sources", "draft the plan", "review and finalize"],
        "what":     ["identify the specific question", "search for evidence", "extract key findings", "formulate answer with citations"],
        "how":      ["identify the specific question", "search for evidence", "extract key findings", "formulate answer with citations"],
        "explore":  ["define exploration boundaries", "gather information on key themes", "identify patterns and connections", "synthesize findings"],
        "investigate": ["define scope", "gather information", "analyze findings", "synthesize and report"],
        "default":  ["understand task", "gather information", "process results", "report"],
    }

    def extract_goal(self, user_input: str, environment_state: str = "", tool_outputs: list = None) -> dict:
        """
        Returns a structured goal dict with:
          goal_text, constraints, subtasks, environment_state, tool_outputs
        """
        constraints = self._extract_constraints(user_input)
        subtasks    = self._extract_subtasks(user_input)

        return {
            "goal_text":         user_input.strip(),
            "constraints":       constraints,
            "subtasks":          subtasks,
            "environment_state": environment_state,
            "tool_outputs":      tool_outputs or [],
        }

    def _extract_constraints(self, text: str) -> list[str]:
        text_lower = text.lower()
        return [label for keyword, label in self.CONSTRAINT_MAP.items() if keyword in text_lower]

    def _extract_subtasks(self, text: str) -> list[str]:
        text_lower = text.lower()
        for key, tasks in self.SUBTASK_TEMPLATES.items():
            if key in text_lower:
                return tasks
        return self.SUBTASK_TEMPLATES["default"]