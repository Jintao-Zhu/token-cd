"""Copy this module and implement it against your robot SDK and calibration.

The example deliberately cannot issue a hardware command. It can be loaded
as a template, but a real adapter must supply observation and action mapping.
"""
from __future__ import annotations

from .adapter import Observation


class ExampleAdapter:
    def observe(self) -> Observation:
        raise NotImplementedError("Read a timestamped RGB frame and robot state")

    def map_action(self, raw_action: dict, observation: Observation) -> dict:
        raise NotImplementedError(
            "Calibrate frame, units, delta convention, gripper and motion limits"
        )

    def send_command(self, command: dict) -> None:
        raise NotImplementedError("Connect the robot SDK only after dry-run review")

    def close(self) -> None:
        pass
