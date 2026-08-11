"""具有 channel masking 的小型 1D Temporal Convolutional Network。"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TCNConfig:
    """固定 epoch 訓練設定；不用 held-out subject 做 early stopping。"""

    hidden_channels: int = 32
    levels: int = 3
    kernel_size: int = 3
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 8
    batch_size: int = 256
    device: str = "cuda"
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.hidden_channels < 1 or self.levels < 1 or self.kernel_size < 2:
            raise ValueError("TCN architecture 參數必須為正數，kernel_size 至少為 2")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs 與 batch_size 必須為正數")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必須介於 0 與 1 之間")


class _CausalConv1d(nn.Conv1d):
    """Conv1d 只在左側保留有效的 causal padding。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
    ) -> None:
        self._causal_padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self._causal_padding,
        )

    def forward(self, values: Tensor) -> Tensor:
        convolved = super().forward(values)
        if self._causal_padding:
            return convolved[..., : -self._causal_padding]
        return convolved


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            _CausalConv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            _CausalConv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        return cast(Tensor, values + self.network(values))


class TemporalConvNet(nn.Module):
    """Input shape 為 ``[batch, time, channel]``，mask 為 ``[batch, channel]``。"""

    def __init__(self, input_channels: int, config: TCNConfig, *, output_classes: int = 1) -> None:
        super().__init__()
        if output_classes < 1:
            raise ValueError("output_classes 必須為正數")
        self.output_classes = output_classes
        self.input_projection = nn.Conv1d(input_channels, config.hidden_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                _ResidualBlock(
                    config.hidden_channels,
                    kernel_size=config.kernel_size,
                    dilation=2**level,
                    dropout=config.dropout,
                )
                for level in range(config.levels)
            ]
        )
        self.classifier = nn.Linear(config.hidden_channels, output_classes)

    def forward(self, values: Tensor, channel_mask: Tensor) -> Tensor:
        if values.ndim != 3 or channel_mask.ndim != 2:
            raise ValueError("values 必須是 [B,T,C]，channel_mask 必須是 [B,C]")
        if values.shape[0] != channel_mask.shape[0] or values.shape[2] != channel_mask.shape[1]:
            raise ValueError("values 與 channel_mask shape 不相容")
        # 先清除 NaN/Inf，再於混合 channels 之前套用 mask。
        # 因此被 mask 的 channel 即使被替換為任意數值，輸出仍必須不變。
        clean = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        masked = clean * channel_mask.to(dtype=clean.dtype).unsqueeze(1)
        hidden = self.input_projection(masked.transpose(1, 2))
        hidden = self.blocks(hidden)
        pooled = hidden.mean(dim=-1)
        logits = self.classifier(pooled)
        if self.output_classes == 1:
            logits = logits.squeeze(-1)
        return cast(Tensor, logits)


def _channel_moments(values: FloatArray) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """只用 fit 收到的 train fold 計算 per-channel normalization。"""

    finite = np.isfinite(values)
    count = finite.sum(axis=(0, 1))
    safe = np.where(finite, values, 0.0)
    total = safe.sum(axis=(0, 1))
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    squared = np.where(finite, (values - mean.reshape(1, 1, -1)) ** 2, 0.0).sum(axis=(0, 1))
    variance = np.divide(squared, count, out=np.ones_like(squared), where=count > 0)
    scale = np.sqrt(variance)
    scale[(scale < 1e-8) | ~np.isfinite(scale)] = 1.0
    return mean.astype(np.float64), scale.astype(np.float64), (count > 0)


def _positive_weight(labels: IntArray) -> float:
    negatives = int(np.sum(labels == 0))
    positives = int(np.sum(labels == 1))
    if positives == 0:
        raise ValueError("train fold 沒有 fall samples")
    return negatives / positives


class TCNClassifier:
    """Scikit-learn-like wrapper，供 nested grouped evaluation harness 使用。"""

    def __init__(self, input_channels: int, config: TCNConfig, *, seed: int) -> None:
        self.input_channels = input_channels
        self.config = config
        self.seed = seed
        self.model_: TemporalConvNet | None = None
        self.mean_: FloatArray | None = None
        self.scale_: FloatArray | None = None
        self.train_supported_: NDArray[np.bool_] | None = None
        self.positive_weight_: float | None = None

    def _set_determinism(self) -> None:
        if self.config.deterministic:
            # 必須在任何 CUDA context / cuBLAS handle 建立前設定。
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if self.config.deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)

    def _device(self) -> torch.device:
        requested = torch.device(self.config.device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("config 要求 CUDA，但 torch.cuda.is_available() 為 False")
        return requested

    def _transform(self, values: FloatArray) -> tuple[Tensor, Tensor]:
        if self.mean_ is None or self.scale_ is None or self.train_supported_ is None:
            raise RuntimeError("TCNClassifier 尚未 fit")
        if values.ndim != 3 or values.shape[2] != self.input_channels:
            raise ValueError("features 必須是 [N,T,input_channels]")
        sample_mask = np.any(np.isfinite(values), axis=1)
        mask = sample_mask & self.train_supported_.reshape(1, -1)
        normalized = (values - self.mean_.reshape(1, 1, -1)) / self.scale_.reshape(1, 1, -1)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(normalized.astype(np.float32)), torch.from_numpy(mask)

    def fit(self, features: FloatArray, labels: IntArray) -> TCNClassifier:
        self._set_determinism()
        values = np.asarray(features, dtype=np.float64)
        target = np.asarray(labels, dtype=np.int64)
        self.mean_, self.scale_, self.train_supported_ = _channel_moments(values)
        self.positive_weight_ = _positive_weight(target)
        input_tensor, mask_tensor = self._transform(values)
        target_tensor = torch.from_numpy(target.astype(np.float32))
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(input_tensor, mask_tensor, target_tensor),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        device = self._device()
        model = TemporalConvNet(self.input_channels, self.config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_function = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.positive_weight_, dtype=torch.float32, device=device)
        )
        model.train()
        for _ in range(self.config.epochs):
            for batch_values, batch_mask, batch_target in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_values.to(device), batch_mask.to(device))
                loss = loss_function(logits, batch_target.to(device))
                loss.backward()
                optimizer.step()
        self.model_ = model
        return self

    def predict_proba(self, features: FloatArray) -> FloatArray:
        if self.model_ is None:
            raise RuntimeError("TCNClassifier 尚未 fit")
        values, masks = self._transform(np.asarray(features, dtype=np.float64))
        device = self._device()
        probabilities: list[Tensor] = []
        self.model_.eval()
        with torch.inference_mode():
            for start in range(0, len(values), self.config.batch_size):
                logits = self.model_(
                    values[start : start + self.config.batch_size].to(device),
                    masks[start : start + self.config.batch_size].to(device),
                )
                probabilities.append(torch.sigmoid(logits).cpu())
        positive = torch.cat(probabilities).numpy().astype(np.float64)
        return cast(FloatArray, np.column_stack((1.0 - positive, positive)))


def _balanced_class_weights(labels: IntArray, num_classes: int) -> FloatArray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if len(counts) != num_classes or np.any(counts == 0):
        raise ValueError("train fold 沒有覆蓋所有 classes")
    weights = len(labels) / (num_classes * counts)
    return weights.astype(np.float64)


class MulticlassTCNClassifier(TCNClassifier):
    """11-class TCN wrapper：class weights 只從 fit 收到的 train fold 計算。"""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        config: TCNConfig,
        *,
        seed: int,
    ) -> None:
        super().__init__(input_channels, config, seed=seed)
        self.num_classes = num_classes
        self.class_weights_: FloatArray | None = None

    def fit(self, features: FloatArray, labels: IntArray) -> MulticlassTCNClassifier:
        self._set_determinism()
        values = np.asarray(features, dtype=np.float64)
        target = np.asarray(labels, dtype=np.int64)
        self.mean_, self.scale_, self.train_supported_ = _channel_moments(values)
        self.class_weights_ = _balanced_class_weights(target, self.num_classes)
        input_tensor, mask_tensor = self._transform(values)
        target_tensor = torch.from_numpy(target)
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(input_tensor, mask_tensor, target_tensor),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        device = self._device()
        model = TemporalConvNet(
            self.input_channels,
            self.config,
            output_classes=self.num_classes,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_function = nn.CrossEntropyLoss(
            weight=torch.from_numpy(self.class_weights_.astype(np.float32)).to(device)
        )
        model.train()
        for _ in range(self.config.epochs):
            for batch_values, batch_mask, batch_target in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_values.to(device), batch_mask.to(device))
                loss = loss_function(logits, batch_target.to(device))
                loss.backward()
                optimizer.step()
        self.model_ = model
        return self

    def predict_proba(self, features: FloatArray) -> FloatArray:
        if self.model_ is None:
            raise RuntimeError("MulticlassTCNClassifier 尚未 fit")
        values, masks = self._transform(np.asarray(features, dtype=np.float64))
        device = self._device()
        probabilities: list[Tensor] = []
        self.model_.eval()
        with torch.inference_mode():
            for start in range(0, len(values), self.config.batch_size):
                logits = self.model_(
                    values[start : start + self.config.batch_size].to(device),
                    masks[start : start + self.config.batch_size].to(device),
                )
                probabilities.append(torch.softmax(logits, dim=1).cpu())
        return torch.cat(probabilities).numpy().astype(np.float64)
