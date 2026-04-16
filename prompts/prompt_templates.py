"""
Layer 3 — Prompt Engineering Layer (Llama 3.2 version)

Llama 3.2 is a proper instruction-following model, so prompts can be
clear and structured without worrying about echoing.
"""


def structured_reasoning_prompt(goal: str, context: dict) -> str:
    """
    Main step-generation prompt.
    Tells Llama exactly what phase the agent is in and what to do next.
    """
    steps     = context.get("steps", [])
    subtasks  = context.get("subtasks", [])
    retrieved = context.get("retrieved", [])

    next_subtask = (
        subtasks[len(steps)] if len(steps) < len(subtasks)
        else subtasks[-1] if subtasks
        else "continue toward goal"
    )

    last_step = steps[-1] if steps else "none"
    top_doc   = retrieved[0][:150] if retrieved else "none"

    return (
        f"You are a research agent completing a task step by step.\n\n"
        f"GOAL: {goal}\n"
        f"CURRENT SUBTASK: {next_subtask}\n"
        f"LAST STEP TAKEN: {last_step}\n"
        f"RELEVANT KNOWLEDGE: {top_doc}\n\n"
        f"Write the single next concrete action to complete the current subtask.\n"
        f"Be specific. Use action verbs. One sentence only. No explanation."
    )


def executor_action_prompt(
    goal: str,
    persistent_alignment_context: str,
    previous_work: str,
    current_subtask: str,
    latest_correction_instruction: str = ""
) -> str:
    """Prompt template for the executor/reasoning engine integration."""
    return (
        "You must remain aligned to the original goal. Do not drift to unrelated topics.\n\n"
        f"PRIMARY GOAL:\n{goal}\n\n"
        f"ALIGNMENT GUIDANCE:\n{persistent_alignment_context}\n\n"
        f"PREVIOUS WORK:\n{previous_work or 'None'}\n\n"
        f"CURRENT SUBTASK:\n{current_subtask}\n"
        + (f"\nLATEST CORRECTION INSTRUCTION:\n{latest_correction_instruction}\n" if latest_correction_instruction else "")
        + "\nWrite the next single concrete action that directly advances the current subtask. Stay on topic and keep the action aligned with the original goal."
    )


def goal_anchoring_prompt(goal: str, constraints: list, steps_so_far: list) -> str:
    """Periodic reminder to re-anchor the agent to the original goal."""
    constraint_str = ", ".join(constraints) if constraints else "none"
    last = steps_so_far[-1] if steps_so_far else "none"
    return (
        f"You are a research agent.\n\n"
        f"ORIGINAL GOAL: {goal}\n"
        f"CONSTRAINTS: {constraint_str}\n"
        f"LAST STEP: {last}\n\n"
        f"Write the single next action that directly advances the original goal.\n"
        f"One sentence only. No explanation."
    )


def reflection_prompt(goal: str, current_step: str) -> str:
    """Ask Llama to evaluate whether a step aligns with the goal."""
    step_short = current_step[:120]
    return (
        f"GOAL: {goal}\n"
        f"STEP: {step_short}\n\n"
        f"Does this step directly help achieve the goal? "
        f"Answer with only 'yes' or 'no'."
    )


def replan_prompt(goal: str, last_step: str) -> str:
    """Triggered by the correction module when drift persists."""
    return (
        f"You are a research agent that drifted from its goal.\n\n"
        f"ORIGINAL GOAL: {goal}\n"
        f"PROBLEMATIC STEP: {last_step[:100]}\n\n"
        f"Write one corrected action that directly advances the original goal.\n"
        f"One sentence only. Start with an action verb."
    )