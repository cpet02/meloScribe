"""Source separation.

Rewritten against the Demucs Python API rather than the CLI, which buys three
things the subprocess version could not have:

  - progress callbacks, so a five-minute separation is not a silent hang
  - `shifts` (test-time augmentation) and `overlap`, the two knobs that
    actually trade time for vocal quality
  - `segment`, so a long song cannot exhaust VRAM mid-run

Model default is `htdemucs_ft`, the fine-tuned variant: roughly +0.5dB vocal
SDR over plain `htdemucs` for ~4x the compute, which is the right trade once
the GPU is in use.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

from .cache import Cache

# All four stems are kept: `other` feeds chord estimation, which the pitch
# engine uses as a harmonic prior, and `bass` disambiguates low octaves.
STEM_NAMES = ('drums', 'bass', 'other', 'vocals')

ProgressFn = Callable[[float, str], None]


@dataclass
class StemResult:
    """Where the separated stems ended up, and how they were made."""
    stems: Dict[str, Path]
    model: str
    device: str
    cached: bool
    duration_s: float = 0.0
    meta: Dict = field(default_factory=dict)

    @property
    def vocals(self) -> Path:
        return self.stems['vocals']

    def __getitem__(self, name: str) -> Path:
        return self.stems[name]


def best_device(requested: Optional[str] = None) -> str:
    """Pick a torch device, preferring CUDA when it is genuinely usable."""
    if requested:
        return requested
    try:
        import torch
    except ImportError:
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


@dataclass
class SeparationSettings:
    """Quality/time trade-offs for separation.

    `shifts` is the meaningful one: N random time-shifted passes averaged
    together, costing N times the compute for a real reduction in artefacts.
    On CPU it is not worth it; on a 3060 it is nearly free in wall-clock terms
    relative to how long you would wait anyway.
    """
    model: str = 'htdemucs_ft'
    shifts: int = 1
    overlap: float = 0.25
    segment: Optional[int] = None
    jobs: int = 0

    @classmethod
    def preset(cls, name: str, device: str = 'cpu') -> 'SeparationSettings':
        if name == 'fast':
            return cls(model='htdemucs', shifts=0, overlap=0.1)
        if name == 'balanced':
            return cls(model='htdemucs_ft', shifts=1, overlap=0.25)
        if name == 'max':
            # Only sane on a GPU; on CPU this is an overnight job.
            return cls(model='htdemucs_ft', shifts=5 if device == 'cuda' else 2,
                       overlap=0.5)
        raise ValueError(f"Unknown separation preset: {name!r}")

    def as_params(self) -> Dict:
        return {'model': self.model, 'shifts': self.shifts,
                'overlap': self.overlap, 'segment': self.segment}


class Separator:
    """Demucs wrapper with caching and progress reporting."""

    def __init__(self, settings: Optional[SeparationSettings] = None,
                 device: Optional[str] = None,
                 cache: Optional[Cache] = None):
        self.device = best_device(device)
        self.settings = settings or SeparationSettings()
        self.cache = cache or Cache()
        self._model = None

    def _load_model(self):
        """Load the Demucs bag-of-models, once per Separator."""
        if self._model is not None:
            return self._model

        from demucs.pretrained import get_model

        self._model = get_model(self.settings.model)
        self._model.to(self.device)
        self._model.eval()
        return self._model

    def separate(self, input_path, progress: Optional[ProgressFn] = None,
                 force: bool = False) -> StemResult:
        """Separate an audio file into stems, reusing cached output when valid."""
        input_path = Path(input_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        expected = [f"{name}.wav" for name in STEM_NAMES]
        entry = self.cache.entry(input_path, 'stems',
                                 params=self.settings.as_params(),
                                 expected=expected)

        if entry.hit and not force:
            if progress:
                progress(1.0, 'stems (cached)')
            meta = entry.read_meta()
            return StemResult(
                stems={n: entry.path / f"{n}.wav" for n in STEM_NAMES},
                model=self.settings.model, device=self.device, cached=True,
                duration_s=meta.get('duration_s', 0.0), meta=meta,
            )

        return self._run(input_path, entry, progress)

    def _run(self, input_path: Path, entry, progress: Optional[ProgressFn]) -> StemResult:
        import torch
        from demucs.apply import apply_model
        from demucs.audio import AudioFile, save_audio

        if progress:
            progress(0.0, f"loading {self.settings.model}")
        model = self._load_model()

        if progress:
            progress(0.05, 'reading audio')
        wav = AudioFile(input_path).read(
            streams=0, samplerate=model.samplerate, channels=model.audio_channels)
        duration_s = wav.shape[-1] / model.samplerate

        # Demucs is trained on standardised input; skipping this costs quality.
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std()
        wav = (wav - mean) / (std + 1e-8)

        if progress:
            progress(0.1, f"separating on {self.device}")

        kwargs = dict(
            shifts=self.settings.shifts,
            overlap=self.settings.overlap,
            device=self.device,
            progress=False,
            num_workers=self.settings.jobs,
        )
        if self.settings.segment is not None:
            kwargs['segment'] = self.settings.segment

        with torch.no_grad():
            try:
                sources = apply_model(model, wav[None], **kwargs)[0]
            except torch.cuda.OutOfMemoryError:
                # Retry in shorter chunks rather than failing the whole job -
                # a long track on a 12GB card is the common trigger.
                if progress:
                    progress(0.1, 'out of VRAM, retrying with shorter segments')
                torch.cuda.empty_cache()
                kwargs['segment'] = 7
                sources = apply_model(model, wav[None], **kwargs)[0]

        sources = sources * std + mean

        if progress:
            progress(0.9, 'writing stems')

        # Write to a temp directory and move into place, so an interrupted run
        # can never leave a partial entry that later looks like a cache hit.
        staging = entry.path.with_name(entry.path.name + '.partial')
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)

        written: Dict[str, Path] = {}
        for name, source in zip(model.sources, sources):
            out_path = staging / f"{name}.wav"
            save_audio(source.cpu(), str(out_path), samplerate=model.samplerate)
            written[name] = out_path

        if entry.path.exists():
            shutil.rmtree(entry.path, ignore_errors=True)
        staging.rename(entry.path)

        entry.write_meta({
            'model': self.settings.model,
            'device': self.device,
            'source': str(input_path),
            'duration_s': duration_s,
            'settings': self.settings.as_params(),
            'sources': list(model.sources),
        })

        if progress:
            progress(1.0, 'stems complete')

        return StemResult(
            stems={n: entry.path / f"{n}.wav" for n in written},
            model=self.settings.model, device=self.device, cached=False,
            duration_s=duration_s,
        )


def refine_vocal_stem(vocals_path, output_path=None,
                      denoise: bool = True):
    """Second-pass cleanup on a separated vocal stem.

    Demucs leaves low-level bleed behind - kick thump, cymbal wash, reverb
    tails of other instruments. Bandpassing to the vocal range and taking the
    harmonic component removes most of it. This is what the pitch engine
    should see, not the raw stem.
    """
    from . import audio as audio_mod

    cleaned = audio_mod.prepare_for_pitch(vocals_path, denoise=denoise)
    if output_path is not None:
        audio_mod.write(output_path, cleaned)
    return cleaned


def separate(input_path, preset: str = 'balanced',
             device: Optional[str] = None,
             progress: Optional[ProgressFn] = None,
             force: bool = False) -> StemResult:
    """Convenience entry point: separate one file with a named quality preset."""
    resolved_device = best_device(device)
    separator = Separator(
        settings=SeparationSettings.preset(preset, resolved_device),
        device=resolved_device,
    )
    return separator.separate(input_path, progress=progress, force=force)
