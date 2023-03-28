import math
from abc import ABC
from enum import Enum
from typing import cast, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms

import os, sys
import configs
from pathlib import Path
import neptune.new as neptune
from neptune.new import Run
from neptune.new.types import File

from io_utils import setup_neptune


class IntervalModuleWithWeights(nn.Module, ABC):
    def __init__(self):
        super().__init__()



class IntervalLinear(IntervalModuleWithWeights):
    def __init__(
            self, in_features: int, out_features: int,
            radius_multiplier: float, max_radius: float, bias: bool,
            normalize_shift: bool, normalize_scale: bool,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.radius_multiplier = radius_multiplier
        self.max_radius = max_radius
        self.normalize_shift = normalize_shift
        self.normalize_scale = normalize_scale

        assert self.radius_multiplier > 0
        assert self.max_radius > 0

        self.weight = Parameter(torch.empty((out_features, in_features)))
        self._radius = Parameter(torch.empty((out_features, in_features)))
        self._shift = Parameter(torch.empty((out_features, in_features)), requires_grad=False)
        self._scale = Parameter(torch.empty((out_features, in_features)), requires_grad=False)

        # TODO test and fix so that it still works with bias=False
        if bias:
            self.bias = Parameter(torch.empty(out_features), requires_grad=True)
            self._bias_radius = Parameter(torch.empty_like(self.bias), requires_grad=False)
            self._bias_shift = Parameter(torch.empty_like(self.bias), requires_grad=False)
            self._bias_scale = Parameter(torch.empty_like(self.bias), requires_grad=False)
        else:
            self.bias = None
        self.reset_parameters()

    def radius_transform(self, params: Tensor):
        return (params * torch.tensor(self.radius_multiplier)).clamp(min=RADIUS_MIN, max=self.max_radius + 0.1) #numeryczne

    @property
    def radius(self) -> Tensor:
        return self.radius_transform(self._radius)

    @radius.setter
    def radius(self, new_radius):
        self._radius = Parameter((self._radius + new_radius), requires_grad=False)

    @property
    def bias_radius(self) -> Tensor:
        return self.radius_transform(self._bias_radius)

    @property
    def bias_shift(self) -> Tensor:
        """Contracted interval middle shift (-1, 1)."""
        if self.normalize_shift:
            eps = torch.tensor(1e-8).to(self._bias_shift.device)
            return (self._bias_shift / torch.max(self.bias_radius, eps)).tanh()
        else:
            return self._bias_shift.tanh()

    @property
    def bias_scale(self) -> Tensor:
        """Contracted interval scale (0, 1)."""
        if self.normalize_scale:
            eps = torch.tensor(1e-8).to(self._bias_scale.device)
            scale = (self._bias_scale / torch.max(self.radius, eps)).sigmoid()
        else:
            scale = self._bias_scale.sigmoid()
        return scale * (1.0 - torch.abs(self.bias_shift))

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore
            self._radius.fill_(self.max_radius)
            self._shift.zero_()
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

                self.bias.zero_()
                self._bias_radius.fill_(self.max_radius)
                self._bias_shift.zero_()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        x = x.refine_names("N", "bounds", "features")  # type: ignore
        assert (x.rename(None) >= 0.0).all(), "All input features must be non-negative."  # type: ignore

        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        assert (x_lower <= x_middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (x_middle <= x_upper).all(), "Middle bound must be less than or equal to upper bound."


        w_middle: Tensor = self.weight
        w_lower = self.weight - self.radius
        w_upper = self.weight + self.radius

        # print(f"DDDD ${self.radius}")

        w_lower_pos = w_lower.clamp(min=0)
        w_lower_neg = w_lower.clamp(max=0)
        w_upper_pos = w_upper.clamp(min=0)
        w_upper_neg = w_upper.clamp(max=0)
        # Further splits only needed for numeric stability with asserts
        w_middle_pos = w_middle.clamp(min=0)
        w_middle_neg = w_middle.clamp(max=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x_lower = x_lower.to(device)
        x_middle = x_middle.to(device)
        x_upper = x_upper.to(device)

        lower = x_lower @ w_lower_pos.t() + x_upper @ w_lower_neg.t()
        upper = x_upper @ w_upper_pos.t() + x_lower @ w_upper_neg.t()
        middle = x_middle @ w_middle_pos.t() + x_middle @ w_middle_neg.t()

        if self.bias is not None:
            b_middle = self.bias + self.bias_shift * self.bias_radius
            b_lower = b_middle - self.bias_scale * self.bias_radius
            b_upper = b_middle + self.bias_scale * self.bias_radius
            lower = lower + b_lower
            upper = upper + b_upper
            middle = middle + b_middle

        assert (lower <= middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (middle <= upper).all(), "Middle bound must be less than or equal to upper bound."

        return torch.stack([lower, middle, upper], dim=1).refine_names("N", "bounds", "features")  # type: ignore



class IntervalModel(nn.Module):
    def __init__(self, radius_multiplier: float, max_radius: float):
        super().__init__()

        self._radius_multiplier = radius_multiplier
        self._max_radius = max_radius

    def interval_children(self) -> List[IntervalModuleWithWeights]:
        return [m for m in self.modules() if isinstance(m, IntervalModuleWithWeights)]

    @property
    def radius_multiplier(self):
        return self._radius_multiplier

    @radius_multiplier.setter
    def radius_multiplier(self, value: float):
        self._radius_multiplier = value
        for m in self.interval_children():
            m.radius_multiplier = value

    @property
    def max_radius(self):
        return self._max_radius

    @max_radius.setter
    def max_radius(self, value: float) -> None:
        self._max_radius = value
        for m in self.interval_children():
            m.max_radius = value

    def radius_transform(self, params: Tensor) -> Tensor:
        for m in self.interval_children():
            return m.radius_transform(params)
        raise ValueError("No IntervalNet modules found in model.")



class IntervalMLP(IntervalModel):
    def __init__(
            self,
            input_size: int,
            hidden_dim: int,
            output_classes: int,
            radius_multiplier: float,
            max_radius: float,
            bias: bool,
            normalize_shift: bool,
            normalize_scale: bool,
    ):
        super().__init__(radius_multiplier=radius_multiplier, max_radius=max_radius)

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_classes = output_classes
        self.normalize_shift = normalize_shift
        self.normalize_scale = normalize_scale
        self.output_names = ['fc1', 'fc2']
        self.fc1 = IntervalLinear(
            self.input_size, self.hidden_dim,
            radius_multiplier=radius_multiplier, max_radius=max_radius,
            bias=bias, normalize_shift=normalize_shift, normalize_scale=normalize_scale,
        )
        self.fc2 = IntervalLinear(
            self.hidden_dim, self.hidden_dim,
            radius_multiplier=radius_multiplier, max_radius=max_radius,
            bias=bias, normalize_shift=normalize_shift, normalize_scale=normalize_scale,
        )
        self.last = IntervalLinear(
            self.hidden_dim,
            self.output_classes,
            radius_multiplier=radius_multiplier,
            max_radius=max_radius,
            bias=bias,
            normalize_shift=normalize_shift,
            normalize_scale=normalize_scale,
        )

    def update_radius(self):
        self.last.radius += torch.tensor(self.radius_multiplier)
        self.fc1.radius += torch.tensor(self.radius_multiplier)
        self.fc2.radius += torch.tensor(self.radius_multiplier)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:  # type: ignore
        x = x.refine_names("N", "C", "H", "W")  # type: ignore  # expected input shape
        x = x.rename(None)  # type: ignore  # drop names for unsupported operations
        x = x.flatten(1)  # (N, features)
        x = x.unflatten(1, (1, -1))  # type: ignore  # (N, bounds, features)
        x = x.tile((1, 3, 1))

        x = x.refine_names("N", "bounds", "features")  # type: ignore
        fc1 = F.relu(self.fc1(x))
        fc2 = F.relu(self.fc2(fc1))

        last = self.last(fc2)

        return {
            "fc1": fc1,
            "fc2": fc2,
            "last": last
        }

    @property
    def device(self):
        return self.fc1.weight.device




