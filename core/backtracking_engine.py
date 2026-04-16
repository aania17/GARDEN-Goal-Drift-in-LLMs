"""
Layer 4 — Backtracking Engine
Shown in the diagram between the Correction Module and the Goal Decomposer.

Triggered when:
  - Execution fails (Executor returns a failure signal)
  - Drift persists across consecutive iterations

Behaviour:
  - Revisit the previous plan node
  - Update subtasks
  - Retry from a safe checkpoint
"""


class BacktrackingEngine:
    """
    Prevents cascading reasoning errors by rolling back to a stable state
    and retrying from there.
    """

    MAX_CONSECUTIVE_DRIFT = 2   # trigger backtrack after this many consecutive drifts

    def __init__(self):
        self._consecutive_drifts = 0
        self._checkpoints = []   # stack of (step_index, subtask_index) snapshots

    def record_drift(self) -> None:
        self._consecutive_drifts += 1

    def clear_drift(self) -> None:
        self._consecutive_drifts = 0

    def should_backtrack(self) -> bool:
        return self._consecutive_drifts >= self.MAX_CONSECUTIVE_DRIFT

    def save_checkpoint(self, step_index: int, subtask_index: int, context_snapshot: dict) -> None:
        """Save a snapshot before executing a risky step."""
        self._checkpoints.append({
            "step_index":    step_index,
            "subtask_index": subtask_index,
            "context":       context_snapshot,
        })

    def backtrack(self) -> dict | None:
        """
        Pop the most recent checkpoint and return it.
        Returns None if no checkpoints are available.
        """
        if not self._checkpoints:
            return None
        self._consecutive_drifts = 0
        checkpoint = self._checkpoints.pop()
        print(f"\n🔙 BACKTRACKING to step {checkpoint['step_index']}, subtask {checkpoint['subtask_index']}")
        return checkpoint

    def force_replan(self, goal_data: dict) -> list[str]:
        """
        Return a fresh subtask list when backtracking has no checkpoint to return to.
        """
        print("\n🔙 No checkpoint available — forcing full replan")
        return goal_data.get("subtasks", ["search", "summarize", "categorize", "report"])