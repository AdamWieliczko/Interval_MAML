import math
import os
from abc import ABC
from enum import Enum
from typing import cast, List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions


RADIUS_MIN = 0.


class IntervalModuleWithWeights(nn.Module, ABC):
    def __init__(self):
        super().__init__()

class IntervalLinear(IntervalModuleWithWeights):
    def __init__(
            self, in_features: int, out_features: int,
            max_radius: float, bias: bool, initial_radius: float,
            normalize_shift: bool, normalize_scale: bool, scale_init: float = -5.,
            initial_eps=0.01
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_radius = max_radius
        self.normalize_shift = normalize_shift
        self.normalize_scale = normalize_scale
        self.scale_init = scale_init
        self.initial_radius = initial_radius
        self.eps = initial_eps

        assert self.max_radius > 0

        self.weight = Parameter(torch.empty((out_features, in_features)))
        self._radius = torch.empty(out_features, in_features)
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
        # self.mode: Mode = Mode.VANILLA
        self.reset_parameters()

    def radius_transform(self, params: Tensor):
        return params.clamp(min=RADIUS_MIN, max=self.max_radius + 0.1)

    @property
    def radius(self) -> Tensor:
        return self.radius_transform(self._radius)

    @radius.setter
    def radius(self, new_radius):
        self._radius = Parameter(new_radius, requires_grad=True)

    @property
    def bias_radius(self) -> Tensor:
        return self.radius_transform(self._bias_radius)

    @property
    def shift(self) -> Tensor:
        """Contracted interval middle shift (-1, 1)."""
        if self.normalize_shift:
            eps = torch.tensor(1e-8).to(self._shift.device)
            return (self._shift / torch.max(self.radius, eps)).tanh()
        else:
            return self._shift.tanh()

    @property
    def bias_shift(self) -> Tensor:
        """Contracted interval middle shift (-1, 1)."""
        if self.normalize_shift:
            eps = torch.tensor(1e-8).to(self._bias_shift.device)
            return (self._bias_shift / torch.max(self.bias_radius, eps)).tanh()
        else:
            return self._bias_shift.tanh()

    @property
    def scale(self) -> Tensor:
        """Contracted interval scale (0, 1)."""
        if self.normalize_scale:
            eps = torch.tensor(1e-8).to(self._scale.device)
            scale = (self._scale / torch.max(self.radius, eps)).sigmoid()
        else:
            scale = self._scale.sigmoid()
        return scale * (1.0 - torch.abs(self.shift))

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
            self._radius.fill_(self.initial_radius)
            self._shift.zero_()
            self._scale.fill_(self.scale_init)
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

                self.bias.zero_()
                self._bias_radius.fill_(self.initial_radius)
                self._bias_shift.zero_()
                self._bias_scale.fill_(self.scale_init)


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


class DeIntervaler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.refine_names("N", "bounds", ...)
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        return x_middle


class ReIntervaler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.rename(None)
        tiler = [1] * (len(x.shape) + 1)
        tiler[1] = 3
        x = x.unsqueeze(1).tile(tiler)
        return x


class IntervalModel(nn.Module):
    def __init__(self, max_radius: float):
        super().__init__()
        self._max_radius = max_radius

    def interval_children(self) -> List[IntervalModuleWithWeights]:
        return [m for m in self.modules() if isinstance(m, IntervalModuleWithWeights)]

    def named_interval_children(self) -> List[Tuple[str, IntervalModuleWithWeights]]:
        return [(n, m)
                for n, m in self.named_modules()
                if isinstance(m, IntervalModuleWithWeights)]

    @property
    def max_radius(self):
        return self._max_radius

    @max_radius.setter
    def max_radius(self, value: float) -> None:
        self._max_radius = value
        for m in self.interval_children():
            m.max_radius = value

    def clamp_radii(self) -> None:
        for m in self.interval_children():
            m.clamp_radii()

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
            max_radius: float,
            initial_radius: float,
            bias: bool,
            normalize_shift: bool,
            normalize_scale: bool,
            scale_init: float,
    ):
        super().__init__(max_radius=max_radius)

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_classes = output_classes
        self.normalize_shift = normalize_shift
        self.normalize_scale = normalize_scale
        self.output_names = ['fc1', 'fc2']
        self.fc1 = IntervalLinear(
            self.input_size, self.hidden_dim,
            max_radius=max_radius,
            initial_radius=initial_radius,
            bias=bias, normalize_shift=normalize_shift, normalize_scale=normalize_scale,
            scale_init=scale_init
        )
        self.fc2 = IntervalLinear(
            self.hidden_dim, self.hidden_dim,
            max_radius=max_radius,
            initial_radius=initial_radius,
            bias=bias, normalize_shift=normalize_shift, normalize_scale=normalize_scale,
            scale_init=scale_init,
        )
        self.last = IntervalLinear(
            self.hidden_dim,
            self.output_classes,
            max_radius=max_radius,
            initial_radius=initial_radius,
            bias=bias,
            normalize_shift=normalize_shift,
            normalize_scale=normalize_scale,
            scale_init=scale_init,
        )

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
    


def robust_output(last_output, target, num_classes, device=torch.device('cuda')):
    """Get the robust version of the current output.
    Returns
    -------
    Tensor
        Robust output logits (lower bound for correct class, upper bounds for incorrect classes).
    """
    output_lower, _, output_higher = last_output.unbind("bounds")
    y_oh = F.one_hot(target, num_classes=num_classes)  # type: ignore
    y_oh = y_oh.to(device)
    return torch.where(y_oh.bool(), output_lower.rename(None), output_higher.rename(None))  # type: ignore

def get_loader(data: torch.utils.data.Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0,
               pin: bool = True) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset=data,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       pin_memory=pin,
                                       num_workers=num_workers)

def test_classification(model: torch.nn.Module,
                        data_loader: torch.utils.data.DataLoader,
                        criterion_class: torch.nn.Module,
                        num_classes,
                        batches: int = 0,
                        device: torch.device = torch.device('cpu')) -> Tuple[float, float]:

    criterion = criterion_class(reduction='sum')
    saved_training = model.training
    model.eval()
    with torch.no_grad():
        running_loss_wc, running_loss = 0.0, 0.0
        correct_wc, correct, total = 0, 0, 0
        for batch, (X, y) in enumerate(data_loader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_pred = model(X)

            if isinstance(y_pred, dict):
                # TODO: nie wiem ile tu jest klas, trzeba tu przekazac te informacje
                # ==== Tutaj zmiany ====
                worst_case_pred = robust_output(y_pred['last'], y, num_classes)
                # print(worst_case_pred)
                # print(worst_case_pred.shape)
                # print(worst_case_pred.argmax(-1))
                # print(worst_case_pred.argmax(-1).shape)

                worst_case_acc = (worst_case_pred.argmax(-1) == y).sum().item()
                correct_wc += worst_case_acc
                worst_case_loss = criterion(worst_case_pred, y)
                running_loss_wc += worst_case_loss
                # ==== Tutaj zmiany ====
                y_pred = y_pred['last'][:, 1].squeeze().rename(None)
            y_pred_max = y_pred.argmax(dim=1)


            loss = criterion(y_pred, y)
            running_loss += loss.item()
            correct += (y_pred_max == y).sum().item()
            total += y.size(0)
            if batch >= batches > 0:
                break
    model.train(saved_training)
    # loss, acc
    return running_loss / total, correct / total, running_loss_wc /  total, correct_wc / total



class IntervalMamlModel:
  def __init__(
        self,
        train_data,
        train_eval_data,
        test_data,
        input_size=28*28,
        hidden_dim_size=500,
        num_of_output_classes=10,
        max_radius=0.5,
        initial_radius=0.0001,
        use_bias=True,
        normalize_shift=False,
        normalize_scale=False,
        scale_init=0.,
        d_eps=0.1,
        neptune_initializer=None,
        dataset_name=""
    ) -> None:
      self.train_data = train_data
      self.train_eval_data = train_eval_data
      self.test_data = test_data
      self.num_of_output_classes = num_of_output_classes
      self.d_eps = d_eps
      self.input_size = input_size
      self.hidden_dim_size = hidden_dim_size
      self.max_radius = max_radius
      self.use_bias = use_bias
      self.normalize_shift = normalize_shift
      self.normalize_scale = normalize_scale
      self.scale_init = scale_init
      self.initial_radius = initial_radius
      self.initialize_model()

      self.neptune_initializer = neptune_initializer
      if neptune_initializer:
        neptune_initializer.initialize_run()
        neptune_initializer.run["parameters"] = {
          "max_radius": max_radius,
          "input_size": input_size,
          "hidden_dim_size": hidden_dim_size,
          "num_of_output_classes": num_of_output_classes,
          "use_bias": use_bias,
          "normalize_shift": normalize_shift,
          "normalize_scale": normalize_scale,
          "scale_init": 0.,
          "dataset": dataset_name,
        }

  def initialize_model(self):
      self.model = IntervalMLP(
          input_size=self.input_size,
          hidden_dim=self.hidden_dim_size,
          output_classes=self.num_of_output_classes,
          max_radius=self.max_radius,
          bias=self.use_bias,
          normalize_shift=self.normalize_shift,
          normalize_scale=self.normalize_scale,
          scale_init=self.scale_init,
          initial_radius=self.initial_radius,
      )

  def optimized_loss(loss, worst_case_loss):
      pass

  def update_eps(self, n_iter: int):
      if n_iter > 0 and n_iter % 100 == 0:
        self.model.last.eps = self.model.last.eps + self.d_eps
        self.model.fc1.eps = self.model.fc1.eps + self.d_eps
        self.model.fc2.eps = self.model.fc2.eps + self.d_eps

  def train_and_verify(
      self,
      batch_size=128,
      epochs=20,
      lr = 1e-3,
      criterion_class=nn.CrossEntropyLoss,
      optimizer_class=torch.optim.Adam,
      device=torch.device('cuda'),
    ):
      train_loader = get_loader(self.train_data, batch_size)
      train_eval_loader = get_loader(self.train_eval_data, batch_size)
      test_loader = get_loader(self.test_data, batch_size)
      self.initialize_model()
      self.model.to(device)
      criterion = criterion_class()
      optimizer = optimizer_class(self.model.parameters(), lr=lr)
      for epoch in range(epochs):
        print(f'Epoch {epoch}')
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = self.model(x) # tu doecelowo ma się zmieniać max_radiuce
            # ==== worst_case_loss ====
            num_classes = self.num_of_output_classes
            worst_case_pred = robust_output(y_pred['last'], y, num_classes)
            worst_case_loss = criterion(worst_case_pred, y) # wprst case loss

            y_pred = y_pred['last'][:, 1].squeeze().rename(None) # biore środki bo zwraca w fromacie: lower, midle, upper
            loss = criterion(y_pred, y) # zwykły los

            p = epoch/(epochs - 1)

            optimizer.zero_grad()
            combined_loss = loss * (1-p) + worst_case_loss * p
            combined_loss.backward()
            optimizer.step()
        self.update_eps(epoch)
        print(f'Current radius: {self.model.last.radius}')
        print(f'Train loss: {loss}')
        print(f'Train wc loss: {worst_case_loss}')
        test_loss, acc, loss_wc, acc_wc = test_classification(self.model, test_loader, criterion_class, num_classes, 0, device)
        print(f'Test ACC: {acc}')
        print(f'Test ACC worst case: {acc_wc}')
        print(".............\n")
        if self.neptune_initializer:
          self.neptune_initializer.run["radius_fc1"].log(self.model.fc1.radius)
          self.neptune_initializer.run["radius_fc2"].log(self.model.fc2.radius)
          self.neptune_initializer.run["radius_last"].log(self.model.last.radius)
          self.neptune_initializer.run["train_loss"].log(loss)
          self.neptune_initializer.run["train_loss_wc"].log(worst_case_loss)
          self.neptune_initializer.run["test_loss"].log(test_loss)
          self.neptune_initializer.run["test_wc_loss"].log(loss_wc)
          self.neptune_initializer.run["test_acc"].log(acc)
          self.neptune_initializer.run["test_acc_wc"].log(acc_wc)
      
      if self.neptune_initializer:
        self.neptune_initializer.stop_run()