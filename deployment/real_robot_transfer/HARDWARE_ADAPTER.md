# Hardware adapter contract

Implement `RobotAdapter` from `adapter.py` in an installation-specific Python
module. Run it with `runner.py --adapter module:ClassName`; the default is a
dry-run. `--execute` is the only runner path that invokes `send_command`.
The exact arm, camera and driver are currently unknown, so no hardware-specific
mapping is provided or assumed. The example module intentionally raises on
every hardware operation.

The current checked-in rollout is for SIMPLER. A real-robot adapter should keep the model/method layer separate from the vendor robot driver and implement the following contract.

## Observation path

1. Acquire the designated camera frame and convert it to RGB `uint8`.
2. Apply the exact crop/resize/color convention expected by the OpenVLA processor.
3. Construct the exact prompt template and task instruction.
4. Read the robot state required by the selected OpenVLA policy/action head.
5. Call the VTCG policy and retain the clean and guided diagnostics for each replan.

`observe()` returns `Observation(rgb, timestamp_s, robot_state)`, where
`rgb` is `[H,W,3]` `uint8` in RGB channel order. Record the camera name,
exposure, resolution, timestamp origin, transform, and any crop before this
point. The runtime resizes to 224×224 with `INTER_AREA` and invokes the
checkpoint processor. OpenVLA does not consume `robot_state` in this entry
point; it is passed to `map_action()` and logged for calibration/audit.

## Action path

1. Decode the seven-dimensional OpenVLA action using the selected, embodiment-matched normalization statistics.
2. Convert translation/rotation to the robot controller's frame, units, and delta/absolute convention.
3. Map gripper values to the actual gripper API and range.
4. Apply workspace, velocity, acceleration, and per-step motion limits before sending commands.
5. Respect the robot's control frequency and action horizon; define how a new replan interrupts or replaces the previous command.
6. Expose a hardware emergency stop and a software action-disable switch.

`map_action(raw_action, observation)` receives `world_vector[3]`,
`rotation_delta[3]`, and `open_gripper[1]` after OpenVLA unnormalization.
It must return a serializable, fully bounded command in the driver frame.
The runner does not apply SIMPLER's `action_scale`, gripper sticky rule, or
Euler-to-axis-angle conversion: those are simulator-specific choices.
`send_command(command)` should reject stale observations and unsupported
commands again at the device boundary, and `close()` releases resources.
No one should pass `raw_action` directly to a robot SDK.

## Calibration and records required on the destination machine

- Robot model, end-effector/TCP transform, and control frame definitions.
- Camera intrinsics/extrinsics and which camera view feeds the model.
- Gripper open/close calibration and action sign/range.
- Per-embodiment action normalization key or fine-tuned checkpoint/statistics.
- Robot-specific prompts, task set, workspace limits, and safe speed limits.
- A dry-run log containing image timestamps, action vectors before/after unnormalization, clipped commands, and robot-state feedback.

The first acceptance check should be image-only and dry-run inference: compute actions, log them, and send no robot command. Next validate bounded motions with an operator present. Simulation success rates alone do not validate the physical action mapping.
