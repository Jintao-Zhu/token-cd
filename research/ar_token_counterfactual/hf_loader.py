"""Load OpenVLA's standalone Hugging Face implementation without training imports."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor


EXPECTED_CODE_HASHES = {
    "configuration_prismatic.py": "68cc5ae34f1b46af3168d8d479cb81bb776965653453fd904aa8eefb6c8f9f68",
    "modeling_prismatic.py": "9ce241c5ca09a4bed73654d0ca509baaff0fc5887d0bdd62e360ca8254f7794a",
    "processing_prismatic.py": "2474a5c3fdd4ff15234924636dc0ead6419c89f49699193be47dbb87510a5425",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_register_openvla_hf(code_dir: str | Path) -> dict[str, type]:
    """Load verified official standalone modules and register their AutoClass mappings."""
    code_dir = Path(code_dir).resolve()
    for filename, expected_hash in EXPECTED_CODE_HASHES.items():
        path = code_dir / filename
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Unexpected OpenVLA code hash for {path}: {actual_hash}")

    package_name = "coreact_openvla_hf"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(code_dir)]
        sys.modules[package_name] = package

    modules = {}
    for stem in ("configuration_prismatic", "processing_prismatic", "modeling_prismatic"):
        module_name = f"{package_name}.{stem}"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, code_dir / f"{stem}.py")
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Unable to load OpenVLA module {module_name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        modules[stem] = module

    config_cls = modules["configuration_prismatic"].OpenVLAConfig
    image_processor_cls = modules["processing_prismatic"].PrismaticImageProcessor
    processor_cls = modules["processing_prismatic"].PrismaticProcessor
    model_cls = modules["modeling_prismatic"].OpenVLAForActionPrediction

    AutoConfig.register("openvla", config_cls, exist_ok=True)
    AutoImageProcessor.register(config_cls, image_processor_cls, exist_ok=True)
    AutoProcessor.register(config_cls, processor_cls, exist_ok=True)
    AutoModelForVision2Seq.register(config_cls, model_cls, exist_ok=True)
    return {
        "config": config_cls,
        "image_processor": image_processor_cls,
        "processor": processor_cls,
        "model": model_cls,
    }
