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
def masked_mse(reconstructed, target, lengths, feature_weights=None):
    timestep_mask = torch.arange(target.size(1), device=target.device).unsqueeze(0)
    timestep_mask = timestep_mask < lengths.to(target.device).unsqueeze(1)
    feature_mask = timestep_mask.unsqueeze(2).expand_as(target)
    squared_errors = (reconstructed - target) ** 2
    return squared_errors[feature_mask].mean(), int(feature_mask.sum().item())


def feature_concept(feature_name):
    if feature_name == "speed" or feature_name.startswith("speed_"):
        return "speed"
    if feature_name.startswith("bearing_"):
        return "bearing"
    if feature_name.startswith("acceleration"):
        return "acceleration"
    return feature_name


def concept_feature_weights(feature_names):
    concepts = [feature_concept(name) for name in feature_names]
    counts = {concept: concepts.count(concept) for concept in set(concepts)}
    return torch.tensor(
        [1.0 / counts[concept] for concept in concepts],
        dtype=torch.float32,
    )


@register_loss("masked_concept_mse")
def masked_concept_mse(reconstructed, target, lengths, feature_weights=None):
    if feature_weights is None:
        raise ValueError("masked_concept_mse requires feature weights")
    timestep_mask = torch.arange(target.size(1), device=target.device).unsqueeze(0)
    timestep_mask = timestep_mask < lengths.to(target.device).unsqueeze(1)
    squared_errors = (reconstructed - target) ** 2
    weights = feature_weights.to(target.device).view(1, 1, -1)
    weighted_sum = (squared_errors * weights * timestep_mask.unsqueeze(2)).sum()
    denominator = timestep_mask.sum() * weights.sum()
    return weighted_sum / denominator, float(denominator.item())
