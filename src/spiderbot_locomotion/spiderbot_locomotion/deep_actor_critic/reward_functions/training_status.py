"""Dataclass for holding training status information."""

from dataclasses import dataclass, field


@dataclass
class TrainingStatus:
    """Class for holding training status information."""

    step_reward: float = 0.0
    episode_reward: float = 0.0
    candidate_reward: float = 0.0
    using_population_training: bool = False
    episode_number: int = 0
    episodes_per_candidate: int = 0
    candidate_number: int = 0
    candidates_per_generation: int = 0
    generation_number: int = 0
    target_reached: bool = False
    episode_terminated: bool = False
    reward_component_labels: list[str] = field(default_factory=list)
    reward_component_values: list[float] = field(default_factory=list)
