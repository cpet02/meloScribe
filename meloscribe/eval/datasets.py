"""Real annotated corpora, converted to the layout `runner --dataset` reads.

    python -m meloscribe.eval.datasets vocadito  ~/data/vocadito  data/eval/vocadito
    python -m meloscribe.eval.datasets ikala     ~/data/iKala     data/eval/ikala
    python -m meloscribe.eval.runner --dataset data/eval/vocadito --systems basic_pitch,ensemble

The layout is one track per stem, in one folder:

    <name>.wav         audio (.mp3 / .flac / .ogg are read too)
    <name>.csv         reference melody f0, `time,frequency` rows, 0 Hz = unvoiced
    <name>.notes.csv   optional reference notes, `onset,offset,midi` rows

A CSV whose stem carries a further dot (`x.notes.csv`, `x.notes.A2.csv`,
`x.alto.csv`) and has no audio of its own is an auxiliary annotation of track
`x`, not a track; with matching audio (`01. Intro.wav`) it is a track.
When `<name>.notes.csv` exists the harness scores note F1 against it instead of
against notes re-derived from the f0 curve, which is the difference between
grading segmentation against a human and grading it against a heuristic.

None of the public corpora ship in this layout - vocadito keeps audio and f0
in different folders under different stems - so each has a converter. Every
converter reproduces the parsing of that corpus's loader in `mirdata`, rather
than re-deriving frame timings or tuning from the papers - except TONAS note
pitches, where mirdata applies the tuning twice (see `convert_tonas`).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .groundtruth import Note, hz_to_midi, midi_to_hz

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg')
NOTES_SUFFIX = '.notes'


def is_auxiliary(csv_path: Path) -> bool:
    """True for `x.notes.csv` and friends: annotations *of* a track."""
    return '.' in Path(csv_path).stem


def notes_path_for(csv_path: Path) -> Path:
    csv_path = Path(csv_path)
    return csv_path.with_name(csv_path.stem + NOTES_SUFFIX + '.csv')


def read_notes_csv(path) -> List[Note]:
    """Read `onset,offset,midi` rows; a header row is skipped."""
    notes: List[Note] = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                onset, offset, midi = (float(v) for v in row[:3])
            except ValueError:
                continue  # header
            if offset > onset:
                notes.append(Note(onset=onset, offset=offset, midi=midi))
    return sorted(notes, key=lambda n: n.onset)


def write_notes_csv(path, notes: Iterable[Note]) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['onset', 'offset', 'midi'])
        for n in sorted(notes, key=lambda n: n.onset):
            writer.writerow([f"{n.onset:.6f}", f"{n.offset:.6f}", f"{n.midi:.4f}"])


def write_f0_csv(path, times, freqs) -> None:
    """Headerless `time,frequency`, the vocadito / MedleyDB convention."""
    times = np.asarray(times, dtype=float)
    freqs = np.nan_to_num(np.asarray(freqs, dtype=float), nan=0.0)
    freqs = np.where(freqs > 0, freqs, 0.0)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        for t, hz in zip(times, freqs):
            f.write(f"{t:.6f},{hz:.4f}\n")


def _hz_notes(onsets, offsets, hz) -> List[Note]:
    midi = hz_to_midi(np.asarray(hz, dtype=float))
    return [Note(onset=float(a), offset=float(b), midi=float(m))
            for a, b, m in zip(onsets, offsets, midi) if b > a and m > 0]


# --- vocadito ---------------------------------------------------------------
# Audio/vocadito_N.wav, Annotations/F0/vocadito_N_f0.csv (time,hz),
# Annotations/Notes/vocadito_N_notesA{1,2}.csv (onset,hz,duration).

def convert_vocadito(src, dst, annotator: str = 'A1') -> List[str]:
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    other = 'A2' if annotator == 'A1' else 'A1'
    names = []
    for audio in sorted((src / 'Audio').glob('vocadito_*.wav')):
        name = audio.stem
        f0 = src / 'Annotations' / 'F0' / f'{name}_f0.csv'
        if not f0.exists():
            continue
        data = np.genfromtxt(f0, delimiter=',', ndmin=2)
        shutil.copy2(audio, dst / f'{name}.wav')
        write_f0_csv(dst / f'{name}.csv', data[:, 0], data[:, 1])
        for who, suffix in ((annotator, NOTES_SUFFIX), (other, f'{NOTES_SUFFIX}.{other}')):
            path = src / 'Annotations' / 'Notes' / f'{name}_notes{who}.csv'
            if path.exists():
                n = np.genfromtxt(path, delimiter=',', ndmin=2)
                write_notes_csv(dst / f'{name}{suffix}.csv',
                                _hz_notes(n[:, 0], n[:, 0] + n[:, 2], n[:, 1]))
        names.append(name)
    return names


# --- iKala ------------------------------------------------------------------
# Wavfile/<id>.wav is stereo: left = accompaniment, right = voice.
# PitchLabel/<id>.pv holds one MIDI value per 32 ms, centred on the frame.

IKALA_STEP = 0.032


def convert_ikala(src, dst) -> List[str]:
    """Write each clip twice: the isolated voice, and voice + accompaniment.

    The pair is the point. The same reference scored on both isolates what the
    accompaniment costs, which a vocal-only corpus cannot measure at all.
    """
    import soundfile as sf

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    names = []
    for wav in sorted((src / 'Wavfile').glob('*.wav')):
        pv = src / 'PitchLabel' / f'{wav.stem}.pv'
        if not pv.exists():
            continue
        midi = np.array([float(line) for line in pv.read_text().split()])
        hz = np.where(midi > 0, midi_to_hz(midi), 0.0)
        times = np.arange(len(midi)) * IKALA_STEP + IKALA_STEP / 2.0

        audio, sr = sf.read(wav, always_2d=True)
        variants = {'voice': audio[:, 1], 'mix': audio[:, 0] + audio[:, 1]}
        for variant, signal in variants.items():
            name = f'{wav.stem}_{variant}'
            # Float, because the channel sum can exceed full scale.
            sf.write(dst / f'{name}.wav', signal, sr, subtype='FLOAT')
            write_f0_csv(dst / f'{name}.csv', times, hz)
            names.append(name)
    return names


# --- TONAS ------------------------------------------------------------------
# <style>/<id>.wav, <id>.f0.Corrected (time, energy, f0 auto, f0 corrected),
# <id>.notes.Corrected: first line the singer's tuning in cents from A440, then
# rows of (onset, duration, midi, energy), where midi is fractional and already
# includes that tuning.

def convert_tonas(src, dst) -> List[str]:
    """TONAS, with note pitches taken as stored.

    The MIDI column already carries the tuning: mirdata's sample file has
    tuning 43 and notes 66.43 and 67.43, and the corrected f0 over the first
    note reads 379.3 Hz - MIDI 66.43 too. mirdata's `_midi_to_hz` adds the
    tuning again, as this converter first did, which puts every note that
    many cents sharp: 43 cents moves a note across the 50-cent scoring
    tolerance. The dataset's own documentation could not be read to settle
    it, so each file's notes are checked against its f0 as they are read.
    """
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    names = []
    for audio in sorted(src.glob('*/*.wav')):
        f0 = audio.with_name(audio.stem + '.f0.Corrected')
        notes = audio.with_name(audio.stem + '.notes.Corrected')
        if not f0.exists():
            continue
        data = np.genfromtxt(f0, ndmin=2)
        shutil.copy2(audio, dst / audio.name)
        write_f0_csv(dst / f'{audio.stem}.csv', data[:, 0], data[:, 3])
        if notes.exists():
            rows = [r for r in csv.reader(notes.read_text().splitlines()) if r]
            reference = [Note(onset=float(r[0]), offset=float(r[0]) + float(r[1]),
                              midi=float(r[2]))
                         for r in rows[1:] if len(r) >= 3 and float(r[1]) > 0]
            _warn_if_notes_leave_the_f0(reference, data[:, 0], data[:, 3],
                                        audio.stem)
            write_notes_csv(dst / f'{audio.stem}{NOTES_SUFFIX}.csv', reference)
        names.append(audio.stem)
    return names


def _warn_if_notes_leave_the_f0(notes: List[Note], times: np.ndarray,
                                hz: np.ndarray, name: str) -> None:
    """Warn when reference notes sit consistently off the corpus's own f0.

    A misread pitch convention (a tuning applied twice, or not at all) shows
    up as every note off by the same cents - which note F1's 50-cent tolerance
    turns into wrong matches without any error.
    """
    offsets = []
    for note in notes:
        inside = (times >= note.onset) & (times < note.offset) & (hz > 0)
        if inside.any():
            offsets.append(100.0 * (float(np.median(hz_to_midi(hz[inside])))
                                    - note.midi))
    if offsets and abs(float(np.median(offsets))) > 25.0:
        import warnings
        warnings.warn(f"{name}: reference notes sit {np.median(offsets):+.0f} "
                      f"cents from the f0 annotation - check the tuning "
                      f"convention before trusting note F1 on this corpus")


# --- MedleyDB-Melody --------------------------------------------------------
# audio/<t>_MIX.wav, melody2/<t>_MELODY2.csv (time,hz), plus
# medleydb_melody_metadata.json carrying `is_instrumental` per track.

def convert_medleydb_melody(src, dst, definition: int = 2,
                            vocal_only: bool = True) -> List[str]:
    """Full mixes with the melody annotation of the given definition.

    `vocal_only` keeps tracks with a singer. Their MELODY2 line can still hand
    over to an instrument for a solo, which is what the corpus defines as
    melody - so a vocal transcriber is expected to lose some frames here.
    """
    if definition not in (1, 2):
        # MELODY3 gives every melodic line its own column. One f0 track per
        # song cannot hold that, and keeping the first column was wrong.
        raise ValueError(f"MedleyDB melody definition {definition} is not a "
                         f"single line; use 1 or 2")
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    meta_path = src / 'medleydb_melody_metadata.json'
    meta: Dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    names = []
    for audio in sorted((src / 'audio').glob('*_MIX.wav')):
        track = audio.name[:-len('_MIX.wav')]
        if vocal_only and meta.get(track, {}).get('is_instrumental', False):
            continue
        ann = src / f'melody{definition}' / f'{track}_MELODY{definition}.csv'
        if not ann.exists():
            continue
        data = np.genfromtxt(ann, delimiter=',', ndmin=2)
        shutil.copy2(audio, dst / f'{track}.wav')
        write_f0_csv(dst / f'{track}.csv', data[:, 0], data[:, 1])
        names.append(track)
    return names


CONVERTERS: Dict[str, Callable[..., List[str]]] = {
    'vocadito': convert_vocadito,
    'ikala': convert_ikala,
    'tonas': convert_tonas,
    'medleydb_melody': convert_medleydb_melody,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Convert a public melody corpus to the runner layout.')
    parser.add_argument('corpus', choices=sorted(CONVERTERS))
    parser.add_argument('src', help='Root of the corpus as distributed')
    parser.add_argument('dst', help='Output folder for --dataset')
    args = parser.parse_args(argv)

    names = CONVERTERS[args.corpus](args.src, args.dst)
    if not names:
        print(f"No annotated tracks found under {args.src}", file=sys.stderr)
        return 1
    print(f"{len(names)} track(s) written to {args.dst}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
