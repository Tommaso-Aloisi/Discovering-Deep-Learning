"""Neural network from scratch (NumPy).

This module is intentionally lightweight and educational:
- Basic layers (Linear), activations (ReLU, Sigmoid)
- Losses (MSE, L1, CrossEntropy)
- A simple training loop (SGD), plus an optional momentum Linear layer

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


class Module:
    """Minimal Torch-like module interface."""

    def __init__(self) -> None:
        self.gradInput = None
        self.output = None

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *input):
        raise NotImplementedError

    def gradientStep(self, lr: float) -> None:
        # Some modules have parameters; others don't.
        return


class MSE(Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes

    def _make_target(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        target = np.zeros((x.shape[0], self.num_classes), dtype=float)
        target[np.arange(x.shape[0]), labels.astype(int)] = 1.0
        return target

    def forward(self, x: np.ndarray, labels: np.ndarray) -> float:
        target = self._make_target(x, labels)
        self.output = np.sum((target - x) ** 2, axis=1)
        return float(np.mean(self.output))

    def backward(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        target = self._make_target(x, labels)
        self.gradInput = 2 * (x - target) / x.shape[0]
        return self.gradInput


class L1(Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes

    def _make_target(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        target = np.zeros((x.shape[0], self.num_classes), dtype=float)
        target[np.arange(x.shape[0]), labels.astype(int)] = 1.0
        return target

    def forward(self, x: np.ndarray, labels: np.ndarray) -> float:
        target = self._make_target(x, labels)
        per_sample = np.sum(np.abs(target - x), axis=1)
        self.output = per_sample
        return float(np.mean(self.output))

    def backward(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        target = self._make_target(x, labels)
        self.gradInput = np.sign(x - target) / x.shape[0]
        return self.gradInput


class CrossEntropy(Module):
    """Cross-entropy on logits (with a stable softmax)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.prob = None

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        # Stable softmax
        z = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(z)
        prob = exp / np.sum(exp, axis=1, keepdims=True)
        self.prob = prob
        n = logits.shape[0]
        loss = -np.log(prob[np.arange(n), labels.astype(int)] + 1e-12)
        self.output = loss
        return float(np.mean(loss))

    def backward(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        assert self.prob is not None
        n = logits.shape[0]
        grad = self.prob.copy()
        grad[np.arange(n), labels.astype(int)] -= 1.0
        grad /= n
        self.gradInput = grad
        return grad


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        # Xavier-ish init
        self.W = np.random.randn(in_features, out_features) / np.sqrt(in_features)
        self.b = np.zeros(out_features) if bias else None

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b) if bias else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = x @ self.W + (self.b if self.b is not None else 0.0)
        return self.output

    def backward(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        self.gradW = x.T @ gradient
        if self.b is not None:
            self.gradb = np.sum(gradient, axis=0)
        self.gradInput = gradient @ self.W.T
        return self.gradInput

    def gradientStep(self, lr: float) -> None:
        self.W -= lr * self.gradW
        if self.b is not None:
            self.b -= lr * self.gradb


class LinearWithMomentum(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.momentum = float(momentum)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b) if self.b is not None else None

    def gradientStep(self, lr: float) -> None:
        self.vW = self.momentum * self.vW + lr * self.gradW
        self.W -= self.vW
        if self.b is not None and self.gradb is not None and self.vb is not None:
            self.vb = self.momentum * self.vb + lr * self.gradb
            self.b -= self.vb


class ReLU(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = np.maximum(0.0, x)
        return self.output

    def backward(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        self.gradInput = gradient * (x > 0)
        return self.gradInput


class Sigmoid(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output

    def backward(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        assert self.output is not None
        self.gradInput = gradient * self.output * (1.0 - self.output)
        return self.gradInput


class MLP(Module):
    """One-hidden-layer MLP."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        non_linearity=ReLU,
        linear=Linear,
    ) -> None:
        super().__init__()
        self.fc1 = linear(in_dim, hidden_dim)
        self.act1 = non_linearity()
        self.fc2 = linear(hidden_dim, out_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.fc1.forward(x)
        h2 = self.act1.forward(h)
        out = self.fc2.forward(h2)
        self.output = out
        return out

    def backward(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        g = self.fc2.backward(self.act1.output, gradient)
        g = self.act1.backward(self.fc1.output, g)
        g = self.fc1.backward(x, g)
        self.gradInput = g
        return g

    def gradientStep(self, lr: float) -> None:
        self.fc2.gradientStep(lr)
        self.fc1.gradientStep(lr)


class DeepMLP(Module):
    """Configurable-depth MLP.

    `layer_dims` is like: [in_dim, h1, h2, ..., out_dim]
    """

    def __init__(
        self,
        layer_dims: Sequence[int],
        non_linearity=ReLU,
        linear=Linear,
    ) -> None:
        super().__init__()
        assert (
            len(layer_dims) >= 2
        ), "layer_dims must include at least [in_dim, out_dim]"
        self.layers: List[Module] = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                self.layers.append(non_linearity())
        self._layer_inputs: List[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._layer_inputs = []
        h = x
        for layer in self.layers:
            self._layer_inputs.append(h)
            h = layer.forward(h)
        self.output = h
        return h

    def backward(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        g = gradient
        for layer, layer_in in zip(reversed(self.layers), reversed(self._layer_inputs)):
            # activations and linear layers share the same signature here
            g = layer.backward(layer_in, g)
        self.gradInput = g
        return g

    def gradientStep(self, lr: float) -> None:
        for layer in self.layers:
            layer.gradientStep(lr)


@dataclass
class TrainHistory:
    train_loss: List[float]
    val_loss: List[float]
    val_acc: List[float]


def evaluate(
    model: Module, loss: Module, data: np.ndarray, labels: np.ndarray
) -> Tuple[float, float]:
    logits = model.forward(data)
    loss_value = loss.forward(logits, labels)
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == labels))
    return float(loss_value), acc


def train_iter(
    model: Module,
    loss: Module,
    batch_data: np.ndarray,
    batch_labels: np.ndarray,
    lr: float,
) -> float:
    logits = model.forward(batch_data)
    train_loss = loss.forward(logits, batch_labels)
    grad_loss = loss.backward(logits, batch_labels)
    model.backward(batch_data, grad_loss)
    model.gradientStep(lr)
    return float(train_loss)


def train(
    model: Module,
    loss: Module,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    lr: float = 1e-2,
    batch_size: int = 64,
    epochs: int = 20,
    shuffle: bool = True,
) -> TrainHistory:
    n = train_data.shape[0]
    hist = TrainHistory(train_loss=[], val_loss=[], val_acc=[])

    for epoch in range(1, epochs + 1):
        if shuffle:
            idx = np.random.permutation(n)
            train_data = train_data[idx]
            train_labels = train_labels[idx]

        # mini-batches
        losses = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            losses.append(
                train_iter(
                    model, loss, train_data[start:end], train_labels[start:end], lr
                )
            )

        val_l, val_acc = evaluate(model, loss, val_data, val_labels)

        hist.train_loss.append(float(np.mean(losses)))
        hist.val_loss.append(val_l)
        hist.val_acc.append(val_acc)

    return hist
