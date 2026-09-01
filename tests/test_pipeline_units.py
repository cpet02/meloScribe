"""Unit tests for the rewritten pipeline's non-audio logic.

Everything here runs without models, network or GPU, so the suite stays fast
enough to run on every change. The audio-accuracy questions belong to the
benchmark harness (`meloscribe.eval.runner`), not to pytest.
"""

import json

import numpy as np
import pytest

from meloscribe.cache import Cache, cache_key, file_fingerprint, params_fingerprint
from meloscribe.key import KeyEstimate, estimate_from_profile
from meloscribe.lyrics.align import (LyricLine, LyricWord, TimedLyrics,
                                     _attach_lines, _attach_words, parse_lrc,
                                     parse_plain, refine_line_times)
from meloscribe.lyrics.lrclib import (LyricsResult, TrackQuery,
                                      metadata_from_filename)
from meloscribe.lyrics.service import LyricsMode, LyricsService, MissingTrackName
from meloscribe.output import format_csv, format_leadsheet, format_lrc, render
from meloscribe.pitch.engine import TranscribedNote, midi_to_name
from meloscribe.pitch.fusion import (FusionSettings, UNVOICED,
                                     build_transition_matrix, fuse_observations,
                                     key_prior_vector, viterbi)
from meloscribe.pitch.grid import N_PITCHES, VoterOutput, gaussian_bump


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def test_cache_key_changes_with_params(tmp_path):
    audio = tmp_path / 'a.mp3'
    audio.write_bytes(b'x' * 4096)

    assert (cache_key(audio, 'stems', {'model': 'htdemucs'})
            != cache_key(audio, 'stems', {'model': 'htdemucs_ft'}))


def test_cache_key_ignores_param_order():
    assert (params_fingerprint({'a': 1, 'b': 2})
            == params_fingerprint({'b': 2, 'a': 1}))


def test_identical_content_under_different_names_shares_a_key(tmp_path):
    """The point of content addressing: the same audio hits the same entry."""
    first, second = tmp_path / 'one.mp3', tmp_path / 'two.mp3'
    first.write_bytes(b'same' * 1000)
    second.write_bytes(b'same' * 1000)
    assert file_fingerprint(first) == file_fingerprint(second)


def test_partial_entry_is_not_a_hit(tmp_path):
    """A crashed run must not leave something that looks cached."""
    audio = tmp_path / 'a.mp3'
    audio.write_bytes(b'x' * 4096)
    cache = Cache(tmp_path / 'cache')

    entry = cache.entry(audio, 'stems', {}, expected=['vocals.wav'])
    assert not entry.hit

    entry.write_meta({'model': 'test'})          # metadata but no audio
    assert not cache.entry(audio, 'stems', {}, expected=['vocals.wav']).hit

    (entry.path / 'vocals.wav').write_bytes(b'\0')
    assert cache.entry(audio, 'stems', {}, expected=['vocals.wav']).hit


# --------------------------------------------------------------------------
# Fusion and decoding
# --------------------------------------------------------------------------

def _voter(name, pitch_index, n_frames=20, strength=1.0, weight=1.0):
    salience = np.full((n_frames, N_PITCHES), 0.01)
    salience[:, pitch_index] = strength
    return VoterOutput(name=name, salience=salience,
                       voicing=np.full(n_frames, 0.9), weight=weight)


def test_agreeing_voters_decode_that_pitch():
    obs = fuse_observations([_voter('a', 24), _voter('b', 24)])
    path = viterbi(obs, build_transition_matrix())
    assert set(path) == {24}


def test_no_single_voter_can_veto():
    """The bug that made the ensemble score below its own best member.

    One voter reporting ~0 for a pitch must not be able to overrule two voters
    that are confident about it.
    """
    settings = FusionSettings()
    dissenter = _voter('dissenter', 40)
    dissenter.salience[:, 24] = 0.0   # actively rejects the majority's pitch

    obs = fuse_observations([_voter('a', 24), _voter('b', 24), dissenter],
                            settings)
    assert set(viterbi(obs, build_transition_matrix(settings))) == {24}


def test_silence_decodes_as_unvoiced():
    n = 15
    quiet = VoterOutput(name='q', salience=np.full((n, N_PITCHES), 0.01),
                        voicing=np.zeros(n))
    path = viterbi(fuse_observations([quiet]), build_transition_matrix())
    assert set(path) == {UNVOICED}


def test_transition_model_bridges_a_corrupted_frame():
    """A single bad frame should not break a sustained note in two."""
    voter = _voter('a', 24, n_frames=30)
    voter.salience[15, 24] = 0.01     # one frame goes wrong
    voter.salience[15, 40] = 1.0

    path = viterbi(fuse_observations([voter]), build_transition_matrix())
    assert path[15] == 24, 'transition model failed to bridge the bad frame'


def test_key_prior_biases_without_filtering():
    """An out-of-key note with strong evidence must still win.

    The old pipeline deleted these outright, taking every accidental with them.
    """
    prior = key_prior_vector([0, 2, 4, 5, 7, 9, 11])   # C major; C#(=1) is out
    out_of_key = 25                                     # C#, an out-of-key pitch

    obs = fuse_observations([_voter('a', out_of_key), _voter('b', out_of_key)],
                            key_prior=prior)
    assert set(viterbi(obs, build_transition_matrix())) == {out_of_key}


def test_gaussian_bump_peaks_at_the_reported_pitch():
    salience = gaussian_bump(np.array([60.0]), np.array([1.0]))
    assert int(np.argmax(salience[0])) == 60 - 36


def test_mismatched_frame_counts_are_rejected():
    with pytest.raises(ValueError, match='frames'):
        fuse_observations([_voter('a', 24, n_frames=10),
                           _voter('b', 24, n_frames=12)])


# --------------------------------------------------------------------------
# Lyrics
# --------------------------------------------------------------------------

def test_parse_lrc_expands_repeated_timestamps():
    lines = parse_lrc('[00:15.00][01:30.25]Chorus line')
    assert [round(l.start, 2) for l in lines] == [15.0, 90.25]
    assert all(l.text == 'Chorus line' for l in lines)


def test_parse_lrc_handles_millisecond_precision():
    assert parse_lrc('[00:12.500]Word')[0].start == pytest.approx(12.5)
    assert parse_lrc('[00:12.50]Word')[0].start == pytest.approx(12.5)


def test_parse_lrc_skips_metadata_and_empty_lines():
    lines = parse_lrc('[ti:Song]\n[ar:Someone]\n[00:05.00]\n[00:10.00]Real')
    assert len(lines) == 1 and lines[0].text == 'Real'


def test_refine_snaps_to_onsets_but_not_beyond_the_limit():
    lines = [LyricLine(start=10.0, text='near'), LyricLine(start=50.0, text='far')]
    refined = refine_line_times(lines, np.array([10.3, 20.0]), max_shift=0.6)

    by_text = {l.text: l.start for l in refined}
    assert by_text['near'] == pytest.approx(10.3)   # snapped
    assert by_text['far'] == pytest.approx(50.0)    # too far, left alone


def test_melisma_gives_several_notes_the_same_word():
    notes = [TranscribedNote(60, 0.0, 0.3, 0.9),
             TranscribedNote(62, 0.3, 0.6, 0.9),
             TranscribedNote(64, 2.0, 2.3, 0.9)]
    _attach_words(notes, [LyricWord('ah', 0.0, 0.7)])

    assert [n.lyric for n in notes] == ['ah', 'ah', None]


def test_line_attachment_respects_line_ends():
    notes = [TranscribedNote(60, 1.0, 1.2, 0.9), TranscribedNote(60, 9.0, 9.2, 0.9)]
    _attach_lines(notes, [LyricLine(start=0.5, text='first', end=2.0)])

    assert notes[0].lyric == 'first'
    assert notes[1].lyric is None, 'note past the line end must not inherit it'


def test_plain_lyrics_are_spread_across_the_duration():
    lines = parse_plain('one\ntwo\nthree', duration=40.0)
    assert len(lines) == 3
    assert all(0 < l.start < 40 for l in lines)


def test_track_name_gate_blocks_and_bypasses():
    with pytest.raises(MissingTrackName):
        LyricsService.require_track_name(TrackQuery(track_name=''),
                                         LyricsMode.ALIGN)
    # The documented bypass must actually bypass.
    LyricsService.require_track_name(None, LyricsMode.OFF)
    LyricsService.require_track_name(TrackQuery(track_name='Song'),
                                     LyricsMode.ALIGN)


def test_duration_mismatch_rejects_a_wrong_version():
    from meloscribe.lyrics.lrclib import LrcLibClient

    query = TrackQuery(track_name='Song', duration=200.0)
    live = LyricsResult(id=1, track_name='Song', artist_name='X', duration=380.0)
    studio = LyricsResult(id=2, track_name='Song', artist_name='X', duration=201.0)

    assert not LrcLibClient._duration_ok(live, query, 3.0)
    assert LrcLibClient._duration_ok(studio, query, 3.0)


def test_filename_metadata_strips_track_numbers():
    query = metadata_from_filename('05 - Radiohead - Karma Police.mp3')
    assert query.track_name == 'Karma Police'
    assert query.artist_name == 'Radiohead'


def test_timed_lyrics_round_trip_through_lrc():
    lyrics = TimedLyrics(lines=[LyricLine(start=75.25, text='Hello')])
    assert '[01:15.25]Hello' in lyrics.to_lrc()


# --------------------------------------------------------------------------
# Key
# --------------------------------------------------------------------------

def test_c_major_profile_is_identified():
    profile = np.zeros(12)
    for pitch_class in (0, 2, 4, 5, 7, 9, 11):
        profile[pitch_class] = 1.0
    profile[0] = 2.0  # tonic emphasis

    key = estimate_from_profile(profile / profile.sum())
    assert key.tonic == 0 and key.is_major


def test_weak_estimate_is_not_used_as_a_prior():
    assert KeyEstimate(tonic=0, is_major=True, confidence=0.2).as_prior() is None
    assert KeyEstimate(tonic=0, is_major=True, confidence=0.9).as_prior() is not None


def test_minor_scale_includes_raised_sixth_and_seventh():
    """Both appear constantly in real melodies; excluding them would penalise
    correct notes as out-of-key."""
    key = KeyEstimate(tonic=9, is_major=False, confidence=1.0)  # A minor
    assert 6 in key.pitch_classes   # F# - raised 6th
    assert 8 in key.pitch_classes   # G# - raised 7th


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def _notes():
    return [TranscribedNote(60, 0.0, 0.5, 0.95, lyric='hel'),
            TranscribedNote(64, 0.5, 1.0, 0.30, lyric='lo')]


def test_midi_to_name_octaves():
    assert midi_to_name(60) == 'C4'
    assert midi_to_name(69) == 'A4'


def test_csv_has_a_row_per_note():
    rows = format_csv(_notes()).strip().splitlines()
    assert len(rows) == 3  # header + 2


def test_json_output_is_valid_and_complete():
    payload = json.loads(render(_notes(), 'json'))
    assert len(payload['notes']) == 2
    assert payload['notes'][0]['note'] == 'C4'


def test_leadsheet_brackets_low_confidence_notes():
    sheet = format_leadsheet(_notes())
    assert '(E4)' in sheet, 'uncertain notes must be visibly marked'
    assert 'C4' in sheet


def test_lrc_export_emits_one_stamp_per_lyric():
    notes = [TranscribedNote(60, 0.0, 0.5, 0.9, lyric='ah'),
             TranscribedNote(62, 0.5, 1.0, 0.9, lyric='ah'),
             TranscribedNote(64, 1.0, 1.5, 0.9, lyric='oh')]
    lines = [l for l in format_lrc(notes).splitlines() if l.startswith('[')]
    assert len(lines) == 2, 'a melisma must not repeat its word every note'


def test_empty_note_list_does_not_crash_any_format():
    for fmt in ('table', 'csv', 'json', 'leadsheet', 'lrc'):
        assert isinstance(render([], fmt), str)


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError, match='Unknown format'):
        render(_notes(), 'sibelius')
