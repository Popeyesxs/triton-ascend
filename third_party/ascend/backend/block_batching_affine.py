# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Integer arithmetic shared by the element and byte domain proofs."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Affine:
    offset: int
    coefficients: tuple = (0, 0, 0, 0)

    @property
    def uniform(self):
        return not any(self.coefficients)

    def add(self, other, sign=1):
        return Affine(self.offset + sign * other.offset,
                      tuple(a + sign * b for a, b in zip(self.coefficients, other.coefficients)))

    def scale(self, scale):
        return Affine(self.offset * scale, tuple(c * scale for c in self.coefficients))

    def bounds(self, counts):
        lo = hi = self.offset
        for coefficient, count in zip(self.coefficients, counts):
            delta = coefficient * (count - 1)
            lo += min(0, delta)
            hi += max(0, delta)
        return lo, hi
