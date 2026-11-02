"""Guidance variants for the Pi0 flow-matching intervention.

Why this file exists
--------------------
Phase-0 diagnosis (`diag_intervention_form.py`) measured how the intervention
changes the *applied* action in the two model families, over existing artifacts:

    discrete argmax (OpenVLA, 1800 episodes / 177k decisions)
        decisions with no change          28.1%
        changed dims per decision         1.55 of 6
        displacement / action norm        0.363
        displacement / natural spread     0.453

    continuous flow (pi0, 1500 episodes / 63k decisions)
        decisions with no change          0.0%
        dims touched per decision         6 of 6
        displacement / action norm        0.0255
        displacement / natural spread     0.0255

So the recipe is *sparse and large* where it works and *dense and tiny* where it
does not.  The dense/tiny profile is what this module lets us change: either
make each application stronger (lambda), make it act on fewer dimensions
(top-k), or make it act at fewer flow steps (step subset).

The original `pi0_shr_policy.py` and `pi0_shr_rollout.py` are deliberately left
untouched so the locked PI0_SHR_5TASK_0_299_V1 results stay reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from research.semantic_token_cd.pi0_shr_policy import Pi0SHRInference


@dataclass(frozen=True)
class GuidanceSpec:
    """How to turn (positive, negative) velocities into a guided velocity."""

    lambd: float = 0.5
    topk_dims: int | None = None          # None = all six guided dimensions
    active_steps: tuple[int, ...] | None = None   # None = every flow step
    rel_gate: float | None = None         # apply only where ||residual||/||positive|| > rel_gate

    def is_active(self, step: int) -> bool:
        return self.active_steps is None or step in self.active_steps

    def describe(self) -> str:
        parts = [f"lambda={self.lambd}"]
        parts.append("all 6 dims" if self.topk_dims is None else f"top-{self.topk_dims} dims by |residual|")
        parts.append("every flow step" if self.active_steps is None else f"flow steps {list(self.active_steps)}")
        if self.rel_gate is not None:
            parts.append(f"relative gate tau={self.rel_gate}")
        return ", ".join(parts)


def variant_guided_velocity(
    positive: torch.Tensor,
    negative: torch.Tensor,
    spec: GuidanceSpec,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Build the guided velocity for one flow step.

    Returns the guided velocity, the raw positive-minus-negative residual, and
    whether the guidance was actually applied at this step (the relative gate can
    suppress it even on a scheduled step).  The last (gripper) dimension always
    stays on the positive branch.
    """
    if positive.shape != negative.shape or positive.ndim != 3:
        raise ValueError("positive and negative velocities must share shape [B,H,A]")
    if positive.shape[-1] < 2 or spec.lambd < 0:
        raise ValueError("variant guidance requires >= 2 action dimensions and lambda >= 0")

    residual = positive.float() - negative.float()
    guided = positive.float().clone()

    active = spec.is_active(step)
    if active and spec.rel_gate is not None:
        # "decisiveness" gate: only intervene where the counterfactual actually
        # disagrees with the clean branch by a meaningful fraction of the clean
        # velocity's own magnitude.  This is the closest flow-matching analogue
        # of an argmax flip, which requires the intervention to overcome a margin.
        guided_dims = residual[..., :-1]
        clean_dims = positive.float()[..., :-1]
        r = float(
            torch.linalg.vector_norm(guided_dims) / torch.clamp(
                torch.linalg.vector_norm(clean_dims), min=1e-9
            )
        )
        active = r > float(spec.rel_gate)

    if active:
        if spec.topk_dims is None:
            guided[..., :-1] += spec.lambd * residual[..., :-1]
        else:
            guided_dims = residual[..., :-1]                       # [B,H,6]
            # rank dimensions by the mean absolute residual over batch and
            # horizon, then keep only the strongest `topk_dims` of them.  This
            # recreates the discrete models' "1.55 of 6 dims changed" profile.
            score = guided_dims.abs().mean(dim=tuple(range(guided_dims.ndim - 1)))
            k = min(int(spec.topk_dims), score.numel())
            if k < 1:
                raise ValueError("topk_dims must be >= 1")
            keep = torch.topk(score, k).indices
            mask = torch.zeros_like(guided_dims)
            mask[..., keep] = 1.0
            guided[..., :-1] += spec.lambd * guided_dims * mask

    if not torch.isfinite(guided).all():
        raise FloatingPointError("non-finite variant-guided velocity")
    return guided.to(positive.dtype), residual, bool(active)


class Pi0VariantInference(Pi0SHRInference):
    """`Pi0SHRInference` with a configurable guidance spec.

    Only `infer` is overridden; every other method is inherited unchanged so the
    clean branch, the semantic negative branch and the flow loop stay identical
    to the locked protocol.  `arm` is accepted for signature compatibility and
    may be any string other than ``pi0_vanilla``.
    """

    def __init__(self, model, tokenizer, *, spec: GuidanceSpec, **kwargs) -> None:
        super().__init__(model, tokenizer, lambd=spec.lambd, **kwargs)
        self.spec = spec

    @torch.inference_mode()
    def infer(
        self,
        inputs: dict[str, torch.Tensor],
        instruction: str,
        *,
        arm: str,
        noise_seed: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        model = self.model
        vanilla = arm == "pi0_vanilla"

        visual = model.multi_modal_projector(model.vision_tower(inputs["pixel_values"]))
        positive = self._condition_branch(inputs, visual)
        selector: dict[str, Any] | None = None
        negative = None
        if not vanilla:
            negative_visual, selector = self._negative_features(visual, instruction)
            negative = self._condition_branch(inputs, negative_visual)

        generator = torch.Generator(device=inputs["pixel_values"].device)
        generator.manual_seed(int(noise_seed))
        action = torch.randn(
            (inputs["pixel_values"].shape[0], model.horizon_steps, model.action_dim),
            generator=generator,
            device=inputs["pixel_values"].device,
            dtype=inputs["pixel_values"].dtype,
        )
        initial_noise = action.detach().float().cpu()
        delta_t = 1.0 / model.num_inference_steps
        time = torch.zeros(action.shape[0], device=action.device, dtype=action.dtype)

        steps = []
        for step in range(model.num_inference_steps):
            positive_velocity = self._velocity(action, time, inputs, positive)
            if negative is None:
                guided_velocity = positive_velocity
                residual = torch.zeros_like(positive_velocity, dtype=torch.float32)
                negative_velocity = None
                applied = False
            else:
                negative_velocity = self._velocity(action, time, inputs, negative)
                guided_velocity, residual, applied = variant_guided_velocity(
                    positive_velocity, negative_velocity, self.spec, step
                )
            action = action + delta_t * guided_velocity
            steps.append({
                "flow_step": step,
                "time": float(time[0].item()),
                "positive_velocity": positive_velocity.detach().to(torch.float16).cpu(),
                "negative_velocity": (
                    None if negative_velocity is None
                    else negative_velocity.detach().to(torch.float16).cpu()
                ),
                "guided_velocity": guided_velocity.detach().to(torch.float16).cpu(),
                "residual_norm": float(torch.linalg.vector_norm(residual[..., :-1]).item()),
                "guidance_active": bool(applied),
                "gripper_velocity_positive_unchanged": bool(
                    torch.equal(guided_velocity[..., -1], positive_velocity[..., -1])
                ),
            })
            time = time + delta_t

        if model.final_action_clip_value is not None:
            action = torch.clamp(
                action, -model.final_action_clip_value, model.final_action_clip_value
            )
        trace = {
            "arm": arm,
            "noise_seed": int(noise_seed),
            "initial_noise": initial_noise,
            "selector": selector,
            "flow_steps": steps,
            "lambda": 0.0 if negative is None else self.spec.lambd,
            "guidance_spec": self.spec.describe(),
            "shared_action_state_between_branches": True,
            "all_gripper_velocity_positive_unchanged": all(
                item["gripper_velocity_positive_unchanged"] for item in steps
            ),
            "reconstruction_finite": (
                selector is None or bool(torch.isfinite(negative.visual_features).all())
            ),
        }
        return action, trace
