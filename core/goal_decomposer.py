"""
Layer 4 — Goal Decomposer
Converts user goal into ordered subtasks for structured execution.
"""

class GoalDecomposer:
    """Decomposes a goal into ordered subtasks."""

    def __init__(self):
        """Initialize goal decomposer."""
        pass

    def decompose(self, goal_text: str, llm=None) -> list:
        """
        Decompose a goal text into ordered subtasks.

        Args:
            goal_text: The user's goal as a string
            llm: Optional LLM instance for advanced decomposition

        Returns:
            List of subtasks (strings)
        """
        # Try to extract subtasks from goal text directly
        if isinstance(goal_text, dict):
            # Handle if called with dict (backward compatibility)
            goal_text = goal_text.get("goal_text", str(goal_text))

        # Use LLM if available for intelligent decomposition
        if llm:
            return self._llm_decompose(goal_text, llm)

        # Default heuristic decomposition based on goal keywords
        return self._heuristic_decompose(goal_text)

    def _llm_decompose(self, goal_text: str, llm) -> list:
        """Use LLM to intelligently decompose goal."""
        prompt = f"Break this goal into 4 ordered subtasks.\nGoal: {goal_text}\nList exactly 4 subtasks, one per line, no numbering."
        try:
            response = llm.generate(prompt, max_length=100)
            tasks = [line.strip() for line in response.split("\n") if line.strip()]
            return tasks[:4] if tasks else self._default_subtasks()
        except Exception as e:
            print(f"Warning: LLM decomposition failed: {e}")
            return self._default_subtasks()

    def _heuristic_decompose(self, goal_text: str) -> list:
        """
        Decompose goal using keyword patterns.
        
        Returns:
            List of 4 ordered subtasks
        """
        goal_lower = goal_text.lower()

        # Literature survey pattern: find → summarize → categorize → report
        if any(keyword in goal_lower for keyword in ["survey", "literature", "review academic", "research papers"]):
            return [
                "Find relevant papers and sources",
                "Summarize key findings and methodologies",
                "Categorize papers by theme or research area",
                "Generate comprehensive report"
            ]

        # Comparative analysis pattern
        elif any(keyword in goal_lower for keyword in ["compare", "comparison", "versus", "vs", "difference between"]):
            return [
                "Define comparison criteria",
                "Gather information on both subjects",
                "Compare across all criteria",
                "Synthesize findings into conclusion"
            ]

        # Analysis/Investigation pattern
        elif any(keyword in goal_lower for keyword in ["analyze", "investigate", "examine", "study"]):
            return [
                "Define analysis framework",
                "Gather relevant data",
                "Apply analysis framework",
                "Document findings and insights"
            ]

        # Planning pattern
        elif any(keyword in goal_lower for keyword in ["plan", "design", "develop strategy", "create"]):
            return [
                "Define objectives and constraints",
                "Research best practices",
                "Develop detailed plan",
                "Validate and refine plan"
            ]

        # Default execution pattern
        return self._default_subtasks()

    @staticmethod
    def _default_subtasks() -> list:
        """Return safe default subtasks."""
        return [
            "Understand the task requirements",
            "Gather necessary information",
            "Process and analyze information",
            "Generate final output"
        ]
