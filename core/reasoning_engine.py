"""
Layer 4 — Reasoning Engine (Llama 3.2 version)

With Llama 3.2 we no longer need heavy prompt sanitisation.
The model follows instructions cleanly.
"""

from prompts.prompt_templates import (
    structured_reasoning_prompt,
    goal_anchoring_prompt,
    reflection_prompt,
    executor_action_prompt,
)


class ReasoningEngine:

    ANCHOR_EVERY_N_STEPS = 3

    # All known prompt-label echoes Llama 3.2 may emit instead of a real action.
    # Checked case-insensitively via .upper() before stripping.
    _ECHO_PREFIXES = [
        "GOAL:",
        "STEP:",
        "ONE SENTENCE:",
        "ACTION:",
        "NEXT SINGLE CONCRETE ACTION:",
        "NEXT CONCRETE ACTION:",
        "NEXT ACTION:",
        "CONCRETE ACTION:",
        "THE NEXT CONCRETE ACTION:",
        "THE NEXT SINGLE CONCRETE ACTION:",
        "HERE IS THE NEXT CONCRETE ACTION:",
        "HERE IS THE NEXT SINGLE CONCRETE ACTION:",
        "WRITE THE NEXT SINGLE CONCRETE ACTION:",
        "WRITE THE NEXT CONCRETE ACTION:",
    ]

    def __init__(self, llm=None):
        """Initialize reasoning engine with optional LLM instance."""
        self.llm = llm

    def reason(
        self,
        prompt: str,
        goal_text: str,
        current_subtask: str,
        previous_work: str,
        persistent_alignment_context: str,
        latest_correction_instruction: str = ""
    ) -> str:
        """
        Generate reasoning/action plan for a subtask using a fully constructed prompt.

        Args:
            prompt: The executor prompt that includes goal, context, and correction guidance
            goal_text: The original goal text
            current_subtask: The current subtask being executed
            previous_work: Past execution context
            persistent_alignment_context: Accumulated correction guidance
            latest_correction_instruction: Most recent correction instruction

        Returns:
            Action plan or reasoning step
        """
        if self.llm is None:
            return self._heuristic_reason(current_subtask)

        try:
            return self.generate_step(prompt)
        except Exception as e:
            print(f"Warning: Reasoning failed: {e}")
            return self._heuristic_reason(current_subtask)

    def generate_step(self, prompt: str) -> str:
        """Generate next reasoning step."""
        if self.llm is None:
            return "Work on the current subtask."

        response = self.llm.generate(prompt, max_length=100)
        cleaned = self._clean(response)

        # Secondary guard: if _clean still returns a known prompt label (e.g. the
        # model emitted only the label with no following content), fall back to
        # heuristic rather than passing a label string downstream.
        if self._is_prompt_label(cleaned):
            return self._heuristic_reason("")

        return cleaned

    def reflect(self, goal_text: str, step: str) -> str:
        """YES/NO alignment check — Llama 3.2 handles this reliably."""
        if self.llm is None:
            keywords = goal_text.lower().split()
            step_lower = step.lower()
            matches = sum(1 for kw in keywords if kw in step_lower)
            return "yes" if matches > 0 else "no"

        prompt = reflection_prompt(goal_text, step)
        result = self.llm.generate(prompt, max_length=10).strip().lower()
        return "yes" if "yes" in result else "no"

    def _heuristic_reason(self, subtask: str) -> str:
        """
        Generate action plan using heuristics when LLM unavailable or output is unusable.

        Args:
            subtask: The subtask to reason about

        Returns:
            Action plan string
        """
        action_map = {
            "find":       "Search academic databases and online repositories for relevant sources",
            "search":     "Perform targeted keyword search across academic databases",
            "read":       "Read and extract key information from identified sources",
            "summarize":  "Summarize the main findings and methodologies from collected sources",
            "extract":    "Extract relevant data points and insights from sources",
            "categorize": "Organize and categorize sources by theme or research area",
            "analyze":    "Analyze patterns, relationships, and trends in the data",
            "synthesize": "Synthesize information from multiple sources into a coherent narrative",
            "report":     "Generate a comprehensive structured report of findings",
            "plan":       "Create a detailed step-by-step action plan",
            "gather":     "Collect necessary resources and information from available sources",
            "compare":    "Compare items systematically across defined dimensions",
            "evaluate":   "Assess and evaluate findings against the original goal criteria",
        }

        subtask_lower = subtask.lower()
        for keyword, action in action_map.items():
            if keyword in subtask_lower:
                return action

        return f"Work on: {subtask}" if subtask.strip() else \
               "Search academic databases for relevant papers and sources on the topic"

    def _clean(self, text: str) -> str:
        """
        Clean Llama 3.2 output. Strips known prompt-label echoes, then
        takes only the first complete sentence or line.
        """
        result = text.strip()

        # Strip all known echo prefixes (case-insensitive)
        result_upper = result.upper()
        for prefix in self._ECHO_PREFIXES:
            if result_upper.startswith(prefix):
                result = result[len(prefix):].strip()
                result_upper = result.upper()
                # A second prefix may follow (e.g. "NEXT CONCRETE ACTION: ACTION:")
                for inner in self._ECHO_PREFIXES:
                    if result_upper.startswith(inner):
                        result = result[len(inner):].strip()
                        break
                break

        # Take first complete sentence or line
        for delimiter in ["\n", ". ", "! ", "? "]:
            if delimiter in result:
                candidate = result.split(delimiter)[0].strip()
                if len(candidate) > 15:
                    result = candidate
                    break

        # Fallback for empty / too-short output
        if len(result) < 5:
            return "Search academic databases for relevant papers on the topic"

        return result

    def _is_prompt_label(self, text: str) -> bool:
        """
        Return True if text is still just a prompt label with no real content.
        Catches cases where the model emitted only a label and _clean left it intact.
        """
        stripped = text.strip().rstrip(":").upper()
        label_fragments = [
            "NEXT SINGLE CONCRETE ACTION",
            "NEXT CONCRETE ACTION",
            "NEXT ACTION",
            "CONCRETE ACTION",
            "ACTION",
            "STEP",
        ]
        return stripped in label_fragments