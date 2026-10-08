"""UniVLA service for paired Vanilla, L14-logit, and Hidden-CD rollouts."""
from __future__ import annotations

import json
import os
import socket
import traceback

import numpy as np
import torch

from research.semantic_token_cd.univla_l14_model_service import (
    Policy as L14Policy,
    recv_message,
    send_message,
)
from research.semantic_token_cd.univla_hidden_cd_model_service import HiddenPolicy


class FourArmPolicy(HiddenPolicy):
    """Expose the two established action-space arms alongside Hidden-CD."""

    @torch.inference_mode()
    def step(self, image: np.ndarray, arm: str, instruction: str):
        if arm in ("vanilla", "l14_matched"):
            return L14Policy.step(self, image, arm, instruction)
        if arm in ("hidden_cd_025", "hidden_cd_05"):
            return HiddenPolicy.step(self, image, arm, instruction)
        raise ValueError(f"unknown four-arm policy arm: {arm}")


def main():
    sock = os.environ["UNIVLA_HIDDEN_SOCK"]
    policy = FourArmPolicy()
    try:
        os.unlink(sock)
    except FileNotFoundError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock)
    server.listen(1)
    print(json.dumps({
        "model_ready": True,
        "service": "univla_four_arm_hidden_cd",
        "selected_layer": 14,
        "arms": ["vanilla", "l14_matched", "hidden_cd_025", "hidden_cd_05"],
    }), flush=True)
    conn, _ = server.accept()
    with conn:
        while True:
            try:
                head, payload = recv_message(conn)
            except EOFError:
                break
            try:
                cmd = head.get("cmd")
                if cmd == "quit":
                    send_message(conn, {"ok": True})
                    break
                if cmd == "reset":
                    result = policy.reset(head["instruction"], int(head["seed"]))
                elif cmd == "step":
                    image = np.frombuffer(payload, dtype=np.uint8).reshape(head["shape"])
                    result = policy.step(image, head["arm"], head["instruction"])
                else:
                    raise ValueError(f"unknown command: {cmd}")
                send_message(conn, {"ok": True, "result": result})
            except Exception as exc:
                traceback.print_exc()
                send_message(conn, {"ok": False, "error": repr(exc)})
    server.close()
    try:
        os.unlink(sock)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
