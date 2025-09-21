from typing import Tuple

def parse_xyz(text: str) -> Tuple[float, float, float]:
    """
    "x,y,z" 형식 문자열을 (x, y, z) float 튜플로 변환.
    공백 허용, 소수/부호 허용.
    """
    if text is None:
        raise ValueError("Empty input.")

    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError("Input must contain exactly three numbers separated by commas (x,y,z).")

    try:
        x = float(parts[0])
        y = float(parts[1])
        z = float(parts[2])
    except Exception as e:
        raise ValueError("Each of x, y, z must be a valid number.") from e

    return (x, y, z)
