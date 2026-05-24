"""Step Reducer - Controls inference step count for efficiency."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StepConfig:
    """Configuration for step reduction."""
    base_steps: int
    reduced_steps: int
    scheduler_type: str = "EulerDiscreteScheduler"


STEP_CONFIGS = {
    "eco": StepConfig(base_steps=25, reduced_steps=8),
    "balanced": StepConfig(base_steps=25, reduced_steps=16),
    "quality": StepConfig(base_steps=25, reduced_steps=25),
}


class StepReducer:
    """Manages inference step reduction strategies."""

    def __init__(self, preset_name: str = "balanced"):
        self.config = STEP_CONFIGS[preset_name]

    @property
    def num_steps(self) -> int:
        return self.config.reduced_steps

    @property
    def reduction_ratio(self) -> float:
        return 1.0 - (self.config.reduced_steps / self.config.base_steps)

    def estimate_time_savings(self, base_time_per_step: float) -> float:
        """Estimate time saved compared to full steps."""
        steps_saved = self.config.base_steps - self.config.reduced_steps
        return steps_saved * base_time_per_step
