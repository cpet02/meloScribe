"""
Phase 5 (supplement): Chord Track Context
Extracts a per-beat chord timeline from the backing (other) stem using
chroma features. Annotates note events with chord_fit for cross-validation.
"""

import librosa
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

_PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


class ChordTracker:
    """Extracts a beat-level chord timeline from an audio stem."""

    def extract_chord_timeline(self, audio_path: str,
                               beat_times: List[float]) -> List[Dict[str, Any]]:
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"  Extracting chord timeline from: {audio_path.name}")

        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        timeline = self._build_chord_timeline(chroma, sr, beat_times)

        print(f"  Chord timeline: {len(timeline)} beat-level chord entries")
        return timeline

    def _build_chord_timeline(self, chroma: np.ndarray, sr: int,
                              beat_times: List[float]) -> List[Dict[str, Any]]:
        if not beat_times:
            return []

        hop_length = 512
        n_frames = chroma.shape[1]
        timeline = []
        beat_times_list = list(beat_times)

        for i, beat_time in enumerate(beat_times_list):
            start_time = beat_time
            if i + 1 < len(beat_times_list):
                end_time = beat_times_list[i + 1]
            else:
                interval = (beat_times_list[-1] - beat_times_list[-2]
                            if len(beat_times_list) >= 2 else 0.5)
                end_time = beat_time + interval

            start_frame = int(librosa.time_to_frames(start_time, sr=sr, hop_length=hop_length))
            end_frame   = int(librosa.time_to_frames(end_time,   sr=sr, hop_length=hop_length))
            start_frame = max(0, min(start_frame, n_frames - 1))
            end_frame   = max(start_frame + 1, min(end_frame, n_frames))

            beat_chroma = chroma[:, start_frame:end_frame].mean(axis=1)
            top3_indices = np.argsort(beat_chroma)[-3:][::-1]
            chord_notes  = [_PITCH_CLASSES[idx] for idx in top3_indices]

            timeline.append({'time': float(beat_time), 'chord_notes': chord_notes})

        return timeline


def get_chord_timeline(audio_path: str,
                       beat_times: List[float]) -> List[Dict[str, Any]]:
    return ChordTracker().extract_chord_timeline(audio_path, beat_times)
