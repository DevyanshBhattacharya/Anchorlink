"""Device selection and the licence allow-list for the learned components.

Devices are picked automatically: CUDA, then Apple MPS, then CPU.  Nothing in the
pipeline assumes a GPU; the phase-0 model is CPU-only, and every later phase
falls back to CPU at a smaller batch size.

The allow-list is not decoration.  The competition rules require the final model
to be MIT or Apache-2.0 and at most 8B parameters, and the packages are reviewed.
``check_model`` refuses anything not on the list, so a wrong model id fails loudly
at load time instead of quietly in a submission.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    params: float          # billions
    licence: str
    role: str


#: Every model this pipeline is allowed to load.  Parameter counts and licences are
#: from the model cards.  Qwen3-Reranker-8B (8.19B), Qwen3-8B (8.2B) and anything
#: under the Llama or Gemma community licences are deliberately absent.
ALLOWED: Dict[str, ModelSpec] = {
    "BAAI/bge-m3": ModelSpec("BAAI/bge-m3", 0.568, "MIT", "bi-encoder"),
    "BAAI/bge-reranker-v2-m3": ModelSpec("BAAI/bge-reranker-v2-m3", 0.568,
                                         "Apache-2.0", "cross-encoder"),
    "ibm-granite/granite-embedding-278m-multilingual": ModelSpec(
        "ibm-granite/granite-embedding-278m-multilingual", 0.278, "Apache-2.0",
        "bi-encoder (fast)"),
    "intfloat/multilingual-e5-small": ModelSpec("intfloat/multilingual-e5-small",
                                                0.118, "MIT", "bi-encoder (tiny)"),
    "microsoft/mdeberta-v3-base": ModelSpec("microsoft/mdeberta-v3-base", 0.280,
                                            "MIT", "cross-encoder (second)"),
    "Qwen/Qwen3-Reranker-4B": ModelSpec("Qwen/Qwen3-Reranker-4B", 4.0, "Apache-2.0",
                                        "LLM judge"),
    "Qwen/Qwen3-Embedding-0.6B": ModelSpec("Qwen/Qwen3-Embedding-0.6B", 0.6,
                                           "Apache-2.0", "bi-encoder (fast)"),
}

MAX_PARAMS_B = 8.0
ALLOWED_LICENCES = {"MIT", "Apache-2.0"}

#: Named here only so a wrong id produces a useful error instead of a quiet rule break.
FORBIDDEN_SUBSTRINGS = ("llama", "gemma", "qwen3-8b", "qwen3-reranker-8b",
                        "mistral", "splade", "deepparse")


def check_model(repo: str) -> ModelSpec:
    """Raise unless ``repo`` is on the allow-list and inside the rules."""
    low = repo.lower()
    for bad in FORBIDDEN_SUBSTRINGS:
        if bad in low:
            raise ValueError(
                f"{repo!r} is excluded: it fails the MIT/Apache-2.0 licence rule or "
                f"the 8B parameter cap (matched {bad!r})")
    spec = ALLOWED.get(repo)
    if spec is None:
        raise ValueError(
            f"{repo!r} is not on the allow-list. Allowed: {sorted(ALLOWED)}")
    if spec.licence not in ALLOWED_LICENCES:
        raise ValueError(f"{repo!r} has licence {spec.licence}, which is not allowed")
    if spec.params > MAX_PARAMS_B:
        raise ValueError(f"{repo!r} has {spec.params}B parameters, over the {MAX_PARAMS_B}B cap")
    return spec


def pick_device(prefer: str | None = None) -> str:
    """``cuda`` -> ``mps`` -> ``cpu``, or whatever ``BER_DEVICE`` says."""
    forced = prefer or os.environ.get("BER_DEVICE")
    if forced:
        return forced
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_info() -> Dict[str, object]:
    info: Dict[str, object] = {"device": pick_device(), "offline": os.environ.get("HF_HUB_OFFLINE")}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except ImportError:
        info["torch"] = None
    return info
