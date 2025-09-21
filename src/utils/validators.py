from typing import Tuple

DEFAULT_MIN = -1e6
DEFAULT_MAX =  1e6

def validate_xyz(p: Tuple[float, float, float],
                 min_val: float = DEFAULT_MIN,
                 max_val: float = DEFAULT_MAX) -> None:
    """
    범위 검증: 각 좌표가 [min_val, max_val] 안인지 확인.
    유효하지 않으면 ValueError.
    """
    x, y, z = p
    for v in (x, y, z):
        if not (min_val <= v <= max_val):
            raise ValueError(f"Value {v} out of allowed range [{min_val}, {max_val}].")
