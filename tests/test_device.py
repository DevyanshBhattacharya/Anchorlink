"""The licence allow-list must refuse what the rules refuse."""
import pytest

from ber.device import ALLOWED, MAX_PARAMS_B, check_model, pick_device


@pytest.mark.parametrize("repo", sorted(ALLOWED))
def test_allowed_models_pass_and_stay_inside_the_rules(repo):
    spec = check_model(repo)
    assert spec.licence in {"MIT", "Apache-2.0"}
    assert spec.params <= MAX_PARAMS_B


@pytest.mark.parametrize("repo", [
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-Reranker-8B",
    "naver/splade-v3",
    "mistralai/Mistral-7B-v0.3",
])
def test_forbidden_models_are_refused(repo):
    with pytest.raises(ValueError):
        check_model(repo)


def test_unknown_model_is_refused():
    with pytest.raises(ValueError, match="allow-list"):
        check_model("some-org/some-model")


def test_total_parameters_of_the_cascade_stay_under_the_cap():
    """The doc reads the 8B cap conservatively: the whole cascade must fit."""
    cascade = ["BAAI/bge-m3", "BAAI/bge-reranker-v2-m3", "Qwen/Qwen3-Reranker-4B",
               "microsoft/mdeberta-v3-base"]
    assert sum(ALLOWED[r].params for r in cascade) <= MAX_PARAMS_B


def test_pick_device_returns_something_usable():
    assert pick_device() in {"cuda", "mps", "cpu"}
    assert pick_device("cpu") == "cpu"
