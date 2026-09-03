from modeling.framework.base import ModelPlugin


_MODEL_REGISTRY = {}


def register_model(model_cls):
    if not issubclass(model_cls, ModelPlugin):
        raise TypeError("Registered models must inherit from ModelPlugin")
    if not model_cls.model_type:
        raise ValueError("Registered models must define model_type")
    if model_cls.model_type in _MODEL_REGISTRY:
        raise ValueError(f"Model type already registered: {model_cls.model_type}")
    _MODEL_REGISTRY[model_cls.model_type] = model_cls
    return model_cls


def create_model_plugin(model_type, config):
    try:
        model_cls = _MODEL_REGISTRY[model_type]
    except KeyError as exc:
        available = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Unknown model type '{model_type}'. Available: {available}") from exc
    return model_cls(config)


def registered_models():
    return {name: model_cls.describe() for name, model_cls in sorted(_MODEL_REGISTRY.items())}

