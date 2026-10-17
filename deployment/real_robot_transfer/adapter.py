"""Typed boundary between the method and a physical robot installation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Observation:
    rgb: np.ndarray
    timestamp_s: float
    robot_state: dict[str, Any]


class RobotAdapter(Protocol):
    """Implement this in a separate, installation-specific module.

    ``map_action`` returns a *fully validated* driver command in the correct
    units and frame. ``send_command`` alone may actuate the robot. The runner
    never interprets OpenVLA's seven action values as hardware commands.
    """

    def observe(self) -> Observation: ...
    def map_action(self, raw_action: dict[str, np.ndarray], observation: Observation) -> dict: ...
    def send_command(self, command: dict) -> None: ...
    def close(self) -> None: ...
