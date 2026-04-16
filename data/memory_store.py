"""
Layer 2 — Goal Memory Layer
Persistent Goal Memory as shown in the architecture diagram.

Stores three things visible in the diagram:
  - Goal Text
  - Goal Embedding (G)
  - Task Constraints (e.g. academic sources, structured summary)

Referenced at every reasoning step to prevent goal loss in long contexts.
"""


class GoalMemory:
    """
    Persistent store for the original task objective.
    Passed into AgentLoop so the goal is always accessible,
    even after many reasoning steps have accumulated in context.
    """

    def __init__(self):
        self._goal_data      = None
        self._goal_embedding = None

    def store_goal(self, goal_data: dict, embedding) -> None:
        """Store the full goal dict and its embedding vector."""
        self._goal_data      = goal_data
        self._goal_embedding = embedding

    # ------------------------------------------------------------------ #
    #  Accessors — raise clearly if called before store_goal()            #
    # ------------------------------------------------------------------ #

    def get_goal(self) -> dict:
        if self._goal_data is None:
            raise RuntimeError("GoalMemory: no goal stored yet. Call store_goal() first.")
        return self._goal_data

    def get_goal_text(self) -> str:
        return self.get_goal()["goal_text"]

    def get_constraints(self) -> list[str]:
        return self.get_goal().get("constraints", [])

    def get_subtasks(self) -> list[str]:
        return self.get_goal().get("subtasks", [])

    def get_embedding(self):
        if self._goal_embedding is None:
            raise RuntimeError("GoalMemory: no embedding stored yet. Call store_goal() first.")
        return self._goal_embedding

    def is_ready(self) -> bool:
        return self._goal_data is not None and self._goal_embedding is not None

    def __repr__(self):
        if not self.is_ready():
            return "GoalMemory(empty)"
        return f"GoalMemory(goal='{self.get_goal_text()[:50]}...', constraints={self.get_constraints()})"