#!/usr/bin/env python3
"""Smoke-load a LIBERO-finetuned OpenVLA checkpoint and decode one action."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from research.ar_token_counterfactual.libero_runtime import load_policy, predict_action

CODE_DIR = Path('/home/leju-suzhou/zjt_ws/token-cd/third_party/openvla/prismatic/extern/hf')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--unnorm-key', required=True)
    p.add_argument('--gpu', type=int, default=2)
    a = p.parse_args()
    torch.cuda.set_device(a.gpu)
    model, processor = load_policy(Path(a.checkpoint), CODE_DIR, device=f'cuda:{a.gpu}')
    image = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    with torch.inference_mode():
        action = predict_action(model, processor, image, 'pick up the black bowl and place it on the plate', unnorm_key=a.unnorm_key)
    print({'checkpoint': str(a.checkpoint), 'unnorm_key': a.unnorm_key,
           'action_shape': tuple(action.shape), 'finite': bool(np.isfinite(action).all()),
           'action': action.tolist()})


if __name__ == '__main__':
    main()
