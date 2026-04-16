"""
Layer 4 — Context Memory
Stores the agent's intermediate state as shown in the diagram:
  - task node       (the current high-level task)
  - subtasks        (decomposed from the goal)
  - observations    (environment feedback)
  - reasoning steps (generated steps so far)
"""


class ContextMemory:
    """
    Lightweight context tree inspired by ReCAP.
    Tracks everything the agent has done during the current loop run.
    """

    def __init__(self):
        self.task_node    = None          # current high-level task description
        self.subtasks     = []            # decomposed subtasks
        self.observations = []            # environment observations
        self.steps        = []            # all reasoning steps (including corrected ones)

    # ------------------------------------------------------------------ #
    #  Writers                                                             #
    # ------------------------------------------------------------------ #

    def set_task_node(self, task: str) -> None:
        self.task_node = task

    def set_subtasks(self, subtasks: list[str]) -> None:
        self.subtasks = list(subtasks)

    def add_step(self, step: str) -> None:
        self.steps.append(step)

    def replace_last_step(self, corrected: str) -> None:
        """Overwrite the last step after drift correction."""
        if self.steps:
            self.steps[-1] = corrected
        else:
            self.steps.append(corrected)

    def add_observation(self, obs: str) -> None:
        self.observations.append(obs)

    # ------------------------------------------------------------------ #
    #  Readers                                                             #
    # ------------------------------------------------------------------ #

    def get_context(self) -> dict:
        return {
            "task_node":    self.task_node,
            "subtasks":     list(self.subtasks),
            "steps":        list(self.steps),
            "observations": list(self.observations),
        }

    def get_last_step(self) -> str | None:
        return self.steps[-1] if self.steps else None

    def step_count(self) -> int:
        return len(self.steps)

    def summary(self) -> str:
        lines = [
            f"Task:        {self.task_node}",
            f"Subtasks:    {self.subtasks}",
            f"Steps taken: {len(self.steps)}",
            f"Observations:{len(self.observations)}",
        ]
        return "\n".join(lines)