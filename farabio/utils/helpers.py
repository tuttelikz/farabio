from typing import Optional

__all__ = ['get_num_parameters', '_make_divisible']


def get_num_parameters(model, only_trainable=False):
    if only_trainable:
        num_model_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    else:
        num_model_params = sum(p.numel() for p in model.parameters())

    return num_model_params


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
