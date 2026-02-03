from __future__ import annotations

from collections.abc import Callable

from .state import PhotometryState

StageFn = Callable[[PhotometryState], PhotometryState]


def run(state: PhotometryState, *stages: StageFn) -> PhotometryState:
    """ Run a sequence of stages on a PhotometryState. """
    for stage in stages:
        state = stage(state)
    return state
