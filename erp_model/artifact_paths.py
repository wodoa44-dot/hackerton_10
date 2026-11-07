"""Resolve an explicit generation model or the deployment's selected version."""
import json
from pathlib import Path


def generation_directory(directory):
    path = Path(directory)
    manifest = path / "manifest.json"
    if manifest.exists():
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if "classification_target" in metadata and "slots" in metadata:
            return path
    if path.name != "run1":
        raise FileNotFoundError("Generation model not found: " + str(path))
    pointer = path.parent / "active_generation.json"
    selected = json.loads(pointer.read_text(encoding="utf-8"))["directory"] if pointer.exists() else "generation_v4"
    target = path.parent / selected
    if not (target / "manifest.json").exists():
        raise FileNotFoundError("Selected generation model is incomplete: " + str(target))
    return target
