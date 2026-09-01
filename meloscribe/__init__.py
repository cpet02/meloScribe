"""meloScribe - vocal melody transcription with multi-voter pitch estimation."""

__version__ = '2.0.0'

from .pipeline import (Pipeline, TranscriptionOutput, TranscriptionRequest,
                       transcribe_file)

__all__ = ['Pipeline', 'TranscriptionRequest', 'TranscriptionOutput',
           'transcribe_file', '__version__']
