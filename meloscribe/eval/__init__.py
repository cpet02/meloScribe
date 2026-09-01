"""Accuracy scoring harness.

Nothing in the transcription pipeline gets tuned without a number attached.
This package defines the ground-truth representation, the datasets we score
against, and the mir_eval-backed metrics themselves.
"""

from .groundtruth import GroundTruth, Prediction, notes_to_f0, f0_to_notes
from .metrics import score, ScoreResult

__all__ = [
    'GroundTruth', 'Prediction', 'notes_to_f0', 'f0_to_notes',
    'score', 'ScoreResult',
]
