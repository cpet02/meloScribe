"""
Tests for pipeline/chord_tracker.py and NoteMapper.annotate_chord_fit().
All tests use mocks — no real audio files required.
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pipeline'))

from chord_tracker import ChordTracker, get_chord_timeline
from note_mapper import NoteMapper

BEAT_TIMES = [0.0, 0.5, 1.0, 1.5]
SAMPLE_RATE = 22050

def _make_chroma(dominant_bins, n_frames=20):
    chroma = np.full((12, n_frames), 0.05)
    for b in dominant_bins:
        chroma[b, :] = 1.0
    return chroma

def _note(note_name, start, end):
    return {'note': note_name, 'start_time': start, 'end_time': end, 'confidence': 0.9}


class TestBuildChordTimeline:

    def setup_method(self):
        self.tracker = ChordTracker()

    def test_returns_one_entry_per_beat(self):
        chroma = _make_chroma([0, 4, 7])
        result = self.tracker._build_chord_timeline(chroma, SAMPLE_RATE, BEAT_TIMES)
        assert len(result) == len(BEAT_TIMES)

    def test_each_entry_has_time_field(self):
        chroma = _make_chroma([0, 4, 7])
        result = self.tracker._build_chord_timeline(chroma, SAMPLE_RATE, BEAT_TIMES)
        for entry in result:
            assert 'time' in entry

    def test_each_entry_has_chord_notes_field(self):
        chroma = _make_chroma([0, 4, 7])
        result = self.tracker._build_chord_timeline(chroma, SAMPLE_RATE, BEAT_TIMES)
        for entry in result:
            assert 'chord_notes' in entry

    def test_time_values_match_beat_times(self):
        chroma = _make_chroma([0, 4, 7])
        result = self.tracker._build_chord_timeline(chroma, SAMPLE_RATE, BEAT_TIMES)
        for entry, bt in zip(result, BEAT_TIMES):
            assert entry['time'] == pytest.approx(bt)

    def test_chord_notes_has_three_elements(self):
        chroma = _make_chroma([0, 4, 7])
        result = self.tracker._build_chord_timeline(chroma, SAMPLE_RATE, BEAT_TIMES)
        for entry in result:
            assert len(entry['chord_notes']) == 3

    def test_chord_notes_are_pitch_class_strings(self):
        valid_pcs = {'C','C#','D','D#','E','F','F#','G','G#','A','A#','B'}
        chroma = _make_chroma([0, 4, 7])
        result = self.tracker._build_chord_timeline(chroma, SAMPLE_RATE, BEAT_TIMES)
        for entry in result:
            for pc in entry['chord_notes']:
                assert pc in valid_pcs

    def test_correct_pitch_classes_detected(self):
        # bins 0=C, 4=E, 7=G
        chroma = _make_chroma([0, 4, 7])
        result = self.tracker._build_chord_timeline(chroma, SAMPLE_RATE, BEAT_TIMES)
        for entry in result:
            assert set(entry['chord_notes']) == {'C', 'E', 'G'}

    def test_empty_beat_times_returns_empty_list(self):
        chroma = _make_chroma([0, 4, 7])
        result = self.tracker._build_chord_timeline(chroma, SAMPLE_RATE, [])
        assert result == []

    def test_single_beat(self):
        chroma = _make_chroma([0, 4, 7])
        result = self.tracker._build_chord_timeline(chroma, SAMPLE_RATE, [0.0])
        assert len(result) == 1


class TestExtractChordTimeline:

    def setup_method(self):
        self.tracker = ChordTracker()

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            self.tracker.extract_chord_timeline('/nonexistent/other.wav', BEAT_TIMES)

    def test_loads_and_returns_timeline(self, tmp_path):
        fake_wav = tmp_path / "other.wav"
        fake_wav.write_bytes(b'\x00' * 44)
        fake_chroma = _make_chroma([0, 4, 7])
        with patch('librosa.load', return_value=(np.zeros(22050), SAMPLE_RATE)), \
             patch('librosa.feature.chroma_cqt', return_value=fake_chroma):
            result = self.tracker.extract_chord_timeline(str(fake_wav), BEAT_TIMES)
        assert len(result) == len(BEAT_TIMES)
        for entry in result:
            assert 'time' in entry and 'chord_notes' in entry

    def test_prints_filename_and_count(self, tmp_path, capsys):
        fake_wav = tmp_path / "other.wav"
        fake_wav.write_bytes(b'\x00' * 44)
        fake_chroma = _make_chroma([0, 4, 7])
        with patch('librosa.load', return_value=(np.zeros(22050), SAMPLE_RATE)), \
             patch('librosa.feature.chroma_cqt', return_value=fake_chroma):
            self.tracker.extract_chord_timeline(str(fake_wav), BEAT_TIMES)
        captured = capsys.readouterr().out
        assert "other.wav" in captured
        assert str(len(BEAT_TIMES)) in captured


class TestGetChordTimelineConvenience:

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            get_chord_timeline('/nonexistent/other.wav', BEAT_TIMES)

    def test_returns_list_of_dicts(self, tmp_path):
        fake_wav = tmp_path / "other.wav"
        fake_wav.write_bytes(b'\x00' * 44)
        fake_chroma = _make_chroma([0, 4, 7])
        with patch('librosa.load', return_value=(np.zeros(22050), SAMPLE_RATE)), \
             patch('librosa.feature.chroma_cqt', return_value=fake_chroma):
            result = get_chord_timeline(str(fake_wav), BEAT_TIMES)
        assert isinstance(result, list)
        for entry in result:
            assert 'time' in entry and 'chord_notes' in entry


class TestAnnotateChordFit:

    def setup_method(self):
        self.mapper = NoteMapper(apply_key_filter=False, smooth=False)

    def test_note_in_chord_true(self):
        timeline = [{'time': 0.0, 'chord_notes': ['C', 'E', 'G']}]
        result = self.mapper.annotate_chord_fit([_note('C4', 0.0, 0.5)], timeline)
        assert result[0]['chord_fit'] is True

    def test_note_not_in_chord_false(self):
        timeline = [{'time': 0.0, 'chord_notes': ['C', 'E', 'G']}]
        result = self.mapper.annotate_chord_fit([_note('D4', 0.0, 0.5)], timeline)
        assert result[0]['chord_fit'] is False

    def test_uses_closest_beat(self):
        # note at 0.6 is closer to beat@0.5 (C,E,G) than beat@1.0 (D,F#,A)
        timeline = [
            {'time': 0.5, 'chord_notes': ['C', 'E', 'G']},
            {'time': 1.0, 'chord_notes': ['D', 'F#', 'A']},
        ]
        result = self.mapper.annotate_chord_fit([_note('C4', 0.6, 0.8)], timeline)
        assert result[0]['chord_fit'] is True

    def test_boundary_closer_to_second_beat(self):
        # note at 0.8 is closer to beat@1.0 (D,F#,A) — C is not in that chord
        timeline = [
            {'time': 0.5, 'chord_notes': ['C', 'E', 'G']},
            {'time': 1.0, 'chord_notes': ['D', 'F#', 'A']},
        ]
        result = self.mapper.annotate_chord_fit([_note('C4', 0.8, 1.0)], timeline)
        assert result[0]['chord_fit'] is False

    def test_empty_timeline_all_false(self):
        notes = [_note('C4', 0.0, 0.5), _note('G4', 0.5, 1.0)]
        result = self.mapper.annotate_chord_fit(notes, [])
        assert all(e['chord_fit'] is False for e in result)

    def test_empty_timeline_no_crash(self):
        result = self.mapper.annotate_chord_fit([_note('C4', 0.0, 0.5)], [])
        assert len(result) == 1

    def test_empty_notes_empty_timeline(self):
        assert self.mapper.annotate_chord_fit([], []) == []

    def test_existing_fields_preserved(self):
        timeline = [{'time': 0.0, 'chord_notes': ['C', 'E', 'G']}]
        note = {'note': 'C4', 'start_time': 0.0, 'end_time': 0.5,
                'confidence': 0.95, 'beat_aligned': True, 'pyin_agrees': False}
        result = self.mapper.annotate_chord_fit([note], timeline)
        assert result[0]['note'] == 'C4'
        assert result[0]['beat_aligned'] is True
        assert result[0]['pyin_agrees'] is False

    def test_strips_octave_digit(self):
        timeline = [{'time': 0.0, 'chord_notes': ['A#', 'C', 'F']}]
        result = self.mapper.annotate_chord_fit([_note('A#4', 0.0, 0.5)], timeline)
        assert result[0]['chord_fit'] is True

    def test_process_adds_chord_fit(self):
        mapper = NoteMapper(confidence_threshold=0.5, apply_key_filter=False,
                            smooth=False, transpose=0)
        pitch_data = [{'time': i*0.05, 'frequency': 261.63, 'confidence': 0.95}
                      for i in range(10)]
        timeline = [{'time': 0.0, 'chord_notes': ['C', 'E', 'G']}]
        result = mapper.process(pitch_data, chord_timeline=timeline)
        assert all('chord_fit' in e for e in result)

    def test_process_without_timeline_no_chord_fit(self):
        mapper = NoteMapper(confidence_threshold=0.5, apply_key_filter=False,
                            smooth=False, transpose=0)
        pitch_data = [{'time': i*0.05, 'frequency': 261.63, 'confidence': 0.95}
                      for i in range(10)]
        result = mapper.process(pitch_data)
        assert all('chord_fit' not in e for e in result)


class TestGetStemPaths:

    def test_raises_when_stems_missing(self, tmp_path):
        from stemmer import get_stem_paths
        # Point at a file whose stems definitely don't exist
        with pytest.raises(FileNotFoundError):
            get_stem_paths(str(tmp_path / "fake_song.mp3"))

    def test_returns_dict_with_vocals_and_other(self, tmp_path):
        from stemmer import get_stem_paths
        from unittest.mock import patch
        import pathlib

        song_name = "test_song"
        base = tmp_path / "separated" / "htdemucs" / song_name
        base.mkdir(parents=True)
        (base / "vocals.wav").write_bytes(b'\x00')
        (base / "other.wav").write_bytes(b'\x00')

        # Patch Path(__file__).parent.parent to point at tmp_path
        fake_file = tmp_path / "pipeline" / "stemmer.py"
        with patch('pipeline.stemmer.Path') as MockPath:
            # Make Path(__file__) return a fake path whose parent.parent = tmp_path
            fake_path_obj = pathlib.Path(str(fake_file))
            MockPath.side_effect = lambda x: (
                fake_path_obj if x == __import__('pipeline.stemmer', fromlist=['stemmer']).__file__
                else pathlib.Path(x)
            )
            # Simpler: just create the real files where get_stem_paths looks
            real_base = pathlib.Path(__file__).parent.parent / "separated" / "htdemucs" / song_name
            real_base.mkdir(parents=True, exist_ok=True)
            (real_base / "vocals.wav").write_bytes(b'\x00')
            (real_base / "other.wav").write_bytes(b'\x00')
            try:
                # Use a fake input path whose stem is song_name
                result = get_stem_paths(f"/fake/path/{song_name}.mp3")
            finally:
                import shutil
                shutil.rmtree(str(real_base), ignore_errors=True)

        assert 'vocals' in result
        assert 'other' in result
        assert result['vocals'].endswith('vocals.wav')
        assert result['other'].endswith('other.wav')
