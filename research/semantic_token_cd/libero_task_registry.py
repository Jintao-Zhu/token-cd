#!/usr/bin/env python3
"""Emit the task registry from the checked-out LIBERO suite (no hand copying)."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def build():
    import ast
    src = Path(__file__).parents[2] / "LIBERO/libero/benchmark/libero_suite_task_map.py"
    if not src.exists():
        src = Path(__file__).parents[3] / "openpi/third_party/libero/libero/libero/benchmark/libero_suite_task_map.py"
    tree = ast.parse(src.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "libero_task_map" for t in n.targets))
    data = ast.literal_eval(node.value)
    return {"spatial": list(data["libero_spatial"]), "object": list(data["libero_object"])}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--list_tasks", choices=("spatial","object")); p.add_argument("--output",type=Path,default=Path("artifacts/libero/task_registry.json")); a=p.parse_args()
    reg=build()
    if a.list_tasks: print("\n".join(reg[a.list_tasks]))
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(reg,indent=2)+"\n")
    print(f"wrote {a.output}")
if __name__ == "__main__": main()
