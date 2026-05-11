"""Optional ML-classifier detector and Hugging Face factory.

The framework's :class:`MLClassifierDetector` is dependency-free: it accepts
any ``Callable[[str], float]`` that maps a payload's text to a probability.
Bring your own classifier (TF-IDF + sklearn, an internal HTTP service, a
local ONNX model, etc.) and you don't need extras.

If you'd like a working Hugging Face setup out of the box::

    pip install soweak[ml]

then::

    from soweak.ml import MLClassifierDetector, transformers_classifier
    detector = MLClassifierDetector(
        classifier=transformers_classifier(
            "protectai/deberta-v3-base-prompt-injection-v2"
        ),
        threshold=0.85,
    )

The bundled factory loads the model at construction (the first inference
is slow; subsequent ones are O(prompt-length)). To run on GPU, pass
``device="cuda"``.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator

from soweak.core.detector import Detector, Signal
from soweak.core.types import Boundary, Context, OwaspCategory, Payload, Severity


class MLClassifierDetector(Detector):
    """Detector that consults a caller-supplied probability function.

    Yields a :class:`Signal` whenever ``classifier(payload.text)`` returns a
    value at or above ``threshold``. The classifier is invoked once per
    payload; pre-tokenisation and batching belong inside the callable.

    Parameters:
      classifier: ``Callable[[str], float]`` returning a probability in
        ``[0, 1]``.
      threshold: minimum probability that produces a signal. Default
        ``0.85``.
      category / severity: passed through into the signal.
      boundaries: which boundary this detector listens on. Default
        ``(Boundary.INPUT,)``.
      name: stable identifier; default ``"ml-classifier"``.
    """

    def __init__(
        self,
        classifier: Callable[[str], float],
        threshold: float = 0.85,
        category: OwaspCategory = OwaspCategory.LLM01_PROMPT_INJECTION,
        severity: Severity = Severity.HIGH,
        boundaries: tuple[Boundary, ...] = (Boundary.INPUT,),
        name: str = "ml-classifier",
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self._classifier = classifier
        self._threshold = threshold
        self._category = category
        self._severity = severity
        self._boundaries = boundaries
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> OwaspCategory:
        return self._category

    @property
    def boundaries(self) -> tuple[Boundary, ...]:
        return self._boundaries

    def inspect(self, payload: Payload, ctx: Context) -> Iterable[Signal]:
        return self._iter(payload)

    def _iter(self, payload: Payload) -> Iterator[Signal]:
        text = payload.text
        if not text:
            return
        prob = float(self._classifier(text))
        if prob < self._threshold:
            return
        yield Signal(
            detector=self._name,
            category=self._category,
            severity=self._severity,
            confidence=prob,
            message=f"ML classifier probability {prob:.3f} >= {self._threshold:.3f}",
            metadata={"threshold": self._threshold, "probability": prob},
        )


# ---------------------------------------------------------------------------
# Hugging Face / transformers factory (optional)
# ---------------------------------------------------------------------------


DEFAULT_HF_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"


def transformers_classifier(
    model: str = DEFAULT_HF_MODEL,
    device: str = "cpu",
    injection_label_index: int = 1,
    max_length: int = 512,
) -> Callable[[str], float]:
    """Build a classifier callable backed by a Hugging Face model.

    Requires ``pip install soweak[ml]``. Loads the model and tokenizer at
    construction time (~hundreds of MB / several seconds on first use,
    depending on the model). The returned callable is thread-safe under
    PyTorch's normal guarantees.

    Parameters:
      model: HF model identifier. Defaults to a public prompt-injection
        classifier — review the model card and its license before
        deploying in production.
      device: ``"cpu"`` (default), ``"cuda"``, ``"mps"``, etc.
      injection_label_index: index of the "injection" / "malicious" class in
        the model's output. Most binary classifiers use ``1``.
      max_length: tokenizer truncation; default ``512``.
    """
    try:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as e:  # pragma: no cover - optional dep
        raise ImportError(
            "MLClassifierDetector + transformers requires "
            "`pip install soweak[ml]` (transformers + torch)."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model)
    classifier_model = AutoModelForSequenceClassification.from_pretrained(model)
    classifier_model.to(device)
    classifier_model.eval()

    def _classify(text: str) -> float:
        if not text:
            return 0.0
        with torch.no_grad():
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)
            outputs = classifier_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            return float(probs[0, injection_label_index].item())

    _classify.model_name = model  # type: ignore[attr-defined]
    return _classify


__all__ = ["DEFAULT_HF_MODEL", "MLClassifierDetector", "transformers_classifier"]
