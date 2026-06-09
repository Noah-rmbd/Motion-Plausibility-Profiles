import torch


_LOSS_REGISTRY = {}


def register_loss(name):
    def decorator(loss_function):
        if name in _LOSS_REGISTRY:
            raise ValueError(f"Loss already registered: {name}")
        _LOSS_REGISTRY[name] = loss_function
        return loss_function

    return decorator


def get_loss(name):
    try:
        return _LOSS_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_LOSS_REGISTRY))
        raise ValueError(f"Unknown loss '{name}'. Available: {available}") from exc


@register_loss("masked_mse")
def masked_mse(reconstructed, target, lengths):
    timestep_mask = torch.arange(target.size(1), device=target.device).unsqueeze(0)
    timestep_mask = timestep_mask < lengths.to(target.device).unsqueeze(1)
    feature_mask = timestep_mask.unsqueeze(2).expand_as(target)
    squared_errors = (reconstructed - target) ** 2
    return squared_errors[feature_mask].mean(), int(feature_mask.sum().item())

