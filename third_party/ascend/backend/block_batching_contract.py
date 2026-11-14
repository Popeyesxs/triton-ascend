# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Per-candidate execution state shared by vector and matrix batching."""
from dataclasses import dataclass, field
from enum import Enum


class LaunchPolicy(Enum):
    OWNED_PLAN = 'owned_plan'
    LEGACY_REPORT = 'legacy_report'


@dataclass(frozen=True, slots=True)
class ExecutionContract:
    _factor: int
    _axis: int
    preserve_grid: bool
    positions: object
    candidate_hash: str
    candidate: object = field(repr=False, compare=False)
    _guard: object = field(repr=False, compare=False)
    policy: LaunchPolicy
    _report: object = field(default=None, repr=False, compare=False)
    _request: object = field(default=None, repr=False, compare=False)

    @property
    def factor(self):
        return self._request.factor if self.policy is LaunchPolicy.LEGACY_REPORT else self._factor

    @property
    def axis(self):
        return self._report.get('selected_axis') if self.policy is LaunchPolicy.LEGACY_REPORT else self._axis

    @classmethod
    def from_validated_candidate(cls, candidate, request, report, positions, preserve_grid):
        from .block_batching import BatchingRejected
        from .block_batching_guard import RuntimeGuard
        owned = report.get('cv_mode') == 'disabled'
        axis = report.get('selected_axis')
        if owned:
            positions = tuple(positions)
            if (report.get('owner') != 'block_batching' or type(axis) is not int or axis not in (0, 1, 2)
                    or report.get('lane_width') != request.factor or type(preserve_grid) is not bool
                    or any(type(p) is not int or p < 0 for p in positions)):
                raise BatchingRejected('VV_EXECUTION_CONTRACT_MISMATCH')
        # RuntimeGuard owns a deep copy and mutable evaluation caches. The
        # existing CV dispatch policy intentionally reads its live public report;
        # moving state into this class must not change that compatibility rule.
        return cls(request.factor, axis, preserve_grid, positions, candidate.hash, candidate,
                   RuntimeGuard(report['guard'],
                                request.factor), LaunchPolicy.OWNED_PLAN if owned else LaunchPolicy.LEGACY_REPORT,
                   None if owned else report, None if owned else request)

    def launch_refusal(self, grid, capture, *, reduces_blocks, batched_blocks, core_floor):
        from .block_batching_launch import candidate_launch_refusal
        from .block_batching import _is_cv_report
        live = self.policy is LaunchPolicy.LEGACY_REPORT
        report = self._report if live else {'selected_axis': self._axis}
        request = self._request if live else self
        return candidate_launch_refusal(request, report, grid, self.preserve_grid, capture,
                                        cv=_is_cv_report(report) if live else False, reduces_blocks=reduces_blocks,
                                        batched_blocks=batched_blocks, core_floor=core_floor)

    def prove(self, arguments, grid):
        return self._guard([arguments[index] for index in self.positions], grid)
