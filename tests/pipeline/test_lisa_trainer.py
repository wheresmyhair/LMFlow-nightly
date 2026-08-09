from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lmflow.pipeline.utils.lisa_trainer import DynamicLayerActivationCallback


class ToyModel(nn.Module):
    def __init__(self, layer_count=2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(layer_count)])


def make_callback(model):
    return DynamicLayerActivationCallback(
        n_layers=1,
        interval_steps=1,
        model=model,
        lisa_layers_attribute="model.layers",
    )


def take_adam_step(optimizer, parameter, gradient):
    optimizer.zero_grad(set_to_none=True)
    parameter.grad = torch.full_like(parameter, gradient)
    optimizer.step()


def test_layer_switch_bounds_state_and_restarts_moments_after_a_gap(monkeypatch):
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, betas=(0.0, 0.5), weight_decay=0.0)
    callback = make_callback(model)
    selections = iter(([0], [1], [0]))
    monkeypatch.setattr(callback, "_sample_active_layers", lambda: next(selections))

    first_parameter = model.layers[0].weight
    second_parameter = model.layers[1].weight

    callback.switch_active_layers(optimizer)
    take_adam_step(optimizer, first_parameter, 1.0)
    assert optimizer.state[first_parameter]["exp_avg_sq"].item() == pytest.approx(0.5)

    callback.switch_active_layers(optimizer)
    assert first_parameter not in optimizer.state
    take_adam_step(optimizer, second_parameter, 3.0)
    assert set(optimizer.state) == {second_parameter}

    callback.switch_active_layers(optimizer)
    assert second_parameter not in optimizer.state
    take_adam_step(optimizer, first_parameter, 2.0)

    # A layer sampled again starts a new local Adam sequence:
    # v_1 = beta_2 * 0 + (1 - beta_2) * g^2.
    assert optimizer.state[first_parameter]["step"].item() == 1
    assert optimizer.state[first_parameter]["exp_avg_sq"].item() == pytest.approx(2.0)
    assert set(optimizer.state) == {first_parameter}


def test_layer_switch_restarts_moments_when_the_same_layer_is_sampled_consecutively(monkeypatch):
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, betas=(0.0, 0.5), weight_decay=0.0)
    callback = make_callback(model)
    monkeypatch.setattr(callback, "_sample_active_layers", lambda: [0])

    parameter = model.layers[0].weight
    callback.switch_active_layers(optimizer)
    take_adam_step(optimizer, parameter, 1.0)
    callback.switch_active_layers(optimizer)
    assert parameter not in optimizer.state
    take_adam_step(optimizer, parameter, 2.0)

    assert optimizer.state[parameter]["step"].item() == 1
    assert optimizer.state[parameter]["exp_avg_sq"].item() == pytest.approx(2.0)


def test_layer_switch_finds_state_through_optimizer_wrapper(monkeypatch):
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    wrapped_optimizer = SimpleNamespace(optimizer=optimizer)
    callback = make_callback(model)
    selections = iter(([0], [1]))
    monkeypatch.setattr(callback, "_sample_active_layers", lambda: next(selections))

    first_parameter = model.layers[0].weight
    callback.switch_active_layers(wrapped_optimizer)
    take_adam_step(optimizer, first_parameter, 1.0)
    callback.switch_active_layers(wrapped_optimizer)

    assert first_parameter not in optimizer.state


def test_train_begin_starts_a_fresh_full_interval_after_resume(monkeypatch):
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    callback = DynamicLayerActivationCallback(
        n_layers=1,
        interval_steps=10,
        model=model,
        lisa_layers_attribute="model.layers",
    )
    selections = iter(([0], [1]))
    monkeypatch.setattr(callback, "_sample_active_layers", lambda: next(selections))
    resumed_state = SimpleNamespace(global_step=13)

    stale_parameter = model.layers[1].weight
    take_adam_step(optimizer, stale_parameter, 1.0)
    callback.on_train_begin(None, resumed_state, None, optimizer=optimizer)

    assert stale_parameter not in optimizer.state
    assert callback.active_layers_indices == [0]
    callback.on_step_begin(None, SimpleNamespace(global_step=22), None, optimizer=optimizer)
    assert callback.active_layers_indices == [0]
    callback.on_step_begin(None, SimpleNamespace(global_step=23), None, optimizer=optimizer)
    assert callback.active_layers_indices == [1]


@pytest.mark.parametrize(
    ("n_layers", "interval_steps"),
    ((0, 1), (3, 1), (1, 0)),
)
def test_invalid_lisa_configuration_fails_early(n_layers, interval_steps):
    with pytest.raises(ValueError):
        DynamicLayerActivationCallback(
            n_layers=n_layers,
            interval_steps=interval_steps,
            model=ToyModel(),
            lisa_layers_attribute="model.layers",
        )
