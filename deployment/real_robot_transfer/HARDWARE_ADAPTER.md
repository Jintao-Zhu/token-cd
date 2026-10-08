# Hardware adapter contract

The current checked-in rollout is for SIMPLER. A real-robot adapter should keep the model/method layer separate from the vendor robot driver and implement the following contract.

## Observation path

1. Acquire the designated camera frame and convert it to RGB `uint8`.
2. Apply the exact crop/resize/color convention expected by the OpenVLA processor.
3. Construct the exact prompt template and task instruction.
4. Read the robot state required by the selected OpenVLA policy/action head.
5. Call the L11-Matched policy and retain the clean and guided diagnostics for each replan.

## Action path

1. Decode the seven-dimensional OpenVLA action using the selected, embodiment-matched normalization statistics.
2. Convert translation/rotation to the robot controller's frame, units, and delta/absolute convention.
3. Map gripper values to the actual gripper API and range.
4. Apply workspace, velocity, acceleration, and per-step motion limits before sending commands.
5. Respect the robot's control frequency and action horizon; define how a new replan interrupts or replaces the previous command.
6. Expose a hardware emergency stop and a software action-disable switch.

## Calibration and records required on the destination machine

- Robot model, end-effector/TCP transform, and control frame definitions.
- Camera intrinsics/extrinsics and which camera view feeds the model.
- Gripper open/close calibration and action sign/range.
- Per-embodiment action normalization key or fine-tuned checkpoint/statistics.
- Robot-specific prompts, task set, workspace limits, and safe speed limits.
- A dry-run log containing image timestamps, action vectors before/after unnormalization, clipped commands, and robot-state feedback.

The first acceptance check should be image-only and dry-run inference: compute actions, log them, and send no robot command. Next validate bounded motions with an operator present. Simulation success rates alone do not validate the physical action mapping.
