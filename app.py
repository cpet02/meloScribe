"""
meloScribe — Streamlit UI
Run with: streamlit run app.py
"""

import sys
import os
import tempfile
import time
import json
from pathlib import Path

import streamlit as st

# Add pipeline to path — must happen before any pipeline imports
_pipeline_dir = str(Path(__file__).parent / "pipeline")
if _pipeline_dir not in sys.path:
    sys.path.insert(0, _pipeline_dir)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="meloScribe",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Pipeline imports ──────────────────────────────────────────────────────────

try:
    from stemmer import isolate_vocals
    from pitch_detector import detect_pitch
    from note_mapper import map_notes, detect_key
    from lyric_aligner import align_lyrics_to_notes
    from output import format_output
    from beat_tracker import get_beat_grid, BeatTracker
    from config import load_config
    _pipeline_error = None
except ImportError as e:
    _pipeline_error = str(e)

# ── Minimal CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    section[data-testid="stSidebar"] { min-width: 300px; max-width: 330px; }
    .stDataFrame { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎵 meloScribe")
    st.caption("Melody transcription for saxophone")
    st.divider()

    cfg = {}
    if not _pipeline_error:
        try:
            cfg = load_config()
        except Exception:
            pass

    st.subheader("Core settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.0, max_value=1.0,
        value=float(cfg.get("confidence", 0.85)),
        step=0.05,
        help="Minimum pitch detection confidence. Higher = fewer but more accurate notes.",
    )

    transpose = st.number_input(
        "Transpose (semitones)",
        min_value=-24, max_value=24,
        value=int(cfg.get("transpose", 9)),
        step=1,
        help="9 = alto sax (Eb). 0 = concert pitch.",
    )

    output_format = st.selectbox(
        "Output format",
        options=["table", "leadsheet", "csv", "json"],
        index=["table", "leadsheet", "csv", "json"].index(cfg.get("format", "table")),
    )

    st.divider()
    st.subheader("Filtering & smoothing")

    use_key_filter = st.toggle(
        "Key-aware filtering",
        value=bool(cfg.get("key_filter", True)),
        help="Remove notes outside the detected key. Eliminates most overtone hits.",
    )

    use_smooth = st.toggle(
        "Note smoothing",
        value=bool(cfg.get("smooth", True)),
        help="Merge flickering same-note pairs and remove short octave outliers.",
    )

    st.divider()
    st.subheader("Cross-validation")

    use_dual_pitch = st.toggle(
        "PYIN cross-validation",
        value=bool(cfg.get("dual_pitch", False)),
        help="Runs a second pitch model and penalises disagreeing notes. Adds ~15–30s.",
    )

    use_chord_context = st.toggle(
        "Chord context",
        value=bool(cfg.get("chord_context", False)),
        help="Annotates notes with chord fit from the backing track.",
    )

    other_wav_path = None
    if use_chord_context:
        other_wav_input = st.text_input(
            "Path to other.wav",
            value="",
            placeholder=r"e.g. C:\...\separated\htdemucs\sample\other.wav",
            help="Full path to the other.wav stem Demucs produced for this song.",
        )
        if other_wav_input.strip():
            p = Path(other_wav_input.strip())
            if p.exists() and p.suffix.lower() == ".wav":
                other_wav_path = str(p)
                st.caption(f"✓ Found: {p.name}")
            else:
                st.warning("File not found or not a .wav — chord context will be skipped.")

    st.divider()
    st.caption("meloScribe — passion project 🎷")

# ── Main area ─────────────────────────────────────────────────────────────────

st.header("Melody Transcription")

if _pipeline_error:
    st.error(f"Pipeline failed to load: {_pipeline_error}\n\nMake sure `pipeline/` contains all modules.")
    st.stop()

# ── File inputs ───────────────────────────────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Audio input")
    input_mode = st.radio(
        "Input type",
        ["MP3 file (full pipeline)", "Pre-stemmed vocals.wav (skip Demucs)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if "MP3" in input_mode:
        audio_file  = st.file_uploader("Upload MP3", type=["mp3"])
        vocals_file = None
    else:
        audio_file  = None
        vocals_file = st.file_uploader("Upload vocals.wav", type=["wav"])

with col_right:
    st.subheader("Lyrics (optional)")
    lyrics_file = st.file_uploader("Upload .lrc file", type=["lrc"])
    if lyrics_file:
        st.caption(f"✓ {lyrics_file.name} loaded")

# ── Run button ────────────────────────────────────────────────────────────────

has_input = audio_file is not None or vocals_file is not None

run_btn = st.button(
    "▶  Transcribe",
    type="primary",
    disabled=not has_input,
    use_container_width=True,
)

if not has_input:
    st.info("Upload an audio file above to get started.")

# ── Pipeline execution ────────────────────────────────────────────────────────

if run_btn and has_input:
    st.divider()
    st.subheader("Progress")

    tmp_dir = tempfile.mkdtemp()
    results    = {}
    pyin_rate  = ""
    chord_rate = ""

    try:
        # Save uploaded files to temp dir
        if audio_file:
            audio_path = os.path.join(tmp_dir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())
            input_path  = audio_path
            vocals_path = None
        else:
            vp = os.path.join(tmp_dir, vocals_file.name)
            with open(vp, "wb") as f:
                f.write(vocals_file.getbuffer())
            input_path  = None
            vocals_path = vp

        lyrics_path = None
        if lyrics_file:
            lyrics_path = os.path.join(tmp_dir, lyrics_file.name)
            with open(lyrics_path, "wb") as f:
                f.write(lyrics_file.getbuffer())

        # ── Phase 1: Stem ────────────────────────────────────────────────────
        with st.status("Phase 1 — Separating vocals…", expanded=True) as s1:
            if vocals_path:
                st.write("Using pre-stemmed vocals — skipping Demucs.")
                s1.update(label="Phase 1 — Vocals ready (pre-stemmed) ✓", state="complete")
            else:
                st.write(f"Running Demucs on `{Path(input_path).name}`…")
                st.write("⏳ This takes 1–3 minutes depending on hardware.")
                t0 = time.time()
                vocals_path = isolate_vocals(input_path)
                elapsed = time.time() - t0
                st.write(f"✓ Isolated in {elapsed:.0f}s → `{Path(vocals_path).name}`")
                s1.update(label=f"Phase 1 — Vocals separated ({elapsed:.0f}s) ✓", state="complete")

        # ── Phase 2: Pitch ───────────────────────────────────────────────────
        with st.status("Phase 2 — Detecting pitch…", expanded=True) as s2:
            t0 = time.time()
            pitch_data = detect_pitch(vocals_path)
            elapsed = time.time() - t0
            st.write(f"✓ {len(pitch_data):,} pitch frames in {elapsed:.1f}s")
            s2.update(label=f"Phase 2 — {len(pitch_data):,} pitch frames ✓", state="complete")

        # ── Phase 2b: Beat grid ──────────────────────────────────────────────
        beat_grid = None
        with st.status("Phase 2b — Beat grid…", expanded=False) as s2b:
            try:
                beat_grid = get_beat_grid(vocals_path)
                s2b.update(
                    label=f"Phase 2b — {beat_grid['bpm']:.1f} BPM, {len(beat_grid['beat_times'])} beats ✓",
                    state="complete",
                )
            except Exception as e:
                s2b.update(label=f"Phase 2b — Beat grid skipped ⚠", state="error")

        # ── Phase 2c: PYIN ───────────────────────────────────────────────────
        pyin_data = None
        if use_dual_pitch:
            with st.status("Phase 2c — PYIN cross-validation…", expanded=True) as s2c:
                try:
                    from pyin_detector import get_pyin_pitch
                    t0 = time.time()
                    pyin_data = get_pyin_pitch(vocals_path)
                    elapsed = time.time() - t0
                    st.write(f"✓ {len(pyin_data):,} voiced frames in {elapsed:.1f}s")
                    s2c.update(label=f"Phase 2c — PYIN: {len(pyin_data):,} frames ✓", state="complete")
                except Exception as e:
                    st.write(f"⚠ {e}")
                    s2c.update(label="Phase 2c — PYIN skipped ⚠", state="error")

        # ── Phase 2d: Chord context ──────────────────────────────────────────
        chord_timeline = None
        if use_chord_context:
            with st.status("Phase 2d — Chord context…", expanded=True) as s2d:
                if not other_wav_path:
                    st.write("⚠ No valid other.wav path set in sidebar.")
                    s2d.update(label="Phase 2d — Chord context skipped (no path) ⚠", state="error")
                else:
                    try:
                        from chord_tracker import get_chord_timeline
                        beat_times = beat_grid["beat_times"] if beat_grid else []
                        chord_timeline = get_chord_timeline(other_wav_path, beat_times)
                        st.write(f"✓ {len(chord_timeline)} chord entries")
                        s2d.update(label=f"Phase 2d — {len(chord_timeline)} chord entries ✓", state="complete")
                    except Exception as e:
                        st.write(f"⚠ {e}")
                        s2d.update(label="Phase 2d — Chord context failed ⚠", state="error")

        # ── Phase 3: Note mapping ────────────────────────────────────────────
        with st.status("Phase 3 — Mapping notes…", expanded=False) as s3:
            note_events = map_notes(
                pitch_data, confidence,
                transpose=transpose,
                apply_key_filter=use_key_filter,
                smooth=use_smooth,
                pyin_data=pyin_data,
                chord_timeline=chord_timeline,
            )
            if beat_grid:
                tracker = BeatTracker()
                note_events = tracker.annotate_beat_alignment(note_events, beat_grid)

            if pyin_data:
                agreed = sum(1 for e in note_events if e.get("pyin_agrees"))
                pyin_rate = f" · PYIN: {agreed}/{len(note_events)} agreed"
            if chord_timeline:
                fit = sum(1 for e in note_events if e.get("chord_fit"))
                chord_rate = f" · Chord fit: {fit}/{len(note_events)}"

            s3.update(
                label=f"Phase 3 — {len(note_events)} note events{pyin_rate}{chord_rate} ✓",
                state="complete",
            )

        # ── Phase 4: Lyrics ──────────────────────────────────────────────────
        with st.status("Phase 4 — Aligning lyrics…", expanded=False) as s4:
            aligned_events = align_lyrics_to_notes(note_events, lyrics_path)
            matched = sum(1 for e in aligned_events if e.get("lyric") is not None)
            label = (f"Phase 4 — Lyrics: {matched}/{len(aligned_events)} matched ✓"
                     if lyrics_path else "Phase 4 — No lyrics provided")
            s4.update(label=label, state="complete")

        key_result  = detect_key(aligned_events)
        output_text = format_output(aligned_events, format_type=output_format)

        results = {
            "aligned_events": aligned_events,
            "key_result":     key_result,
            "output_text":    output_text,
            "note_count":     len(aligned_events),
            "bpm":            beat_grid["bpm"] if beat_grid else None,
        }

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        with st.expander("Full traceback"):
            import traceback
            st.code(traceback.format_exc())

    # ── Results ───────────────────────────────────────────────────────────────

    if results:
        st.divider()
        st.subheader("Results")

        key = results["key_result"]
        candidates_str = "  ·  ".join(f"{k} ({v:.0%})" for k, v in key["candidates"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Notes detected", results["note_count"])
        m2.metric("Estimated key", key["key"])
        m3.metric("Key confidence", f"{key['score']:.0%}")
        m4.metric("Tempo", f"{results['bpm']:.0f} BPM" if results["bpm"] else "—")
        st.caption(f"Key candidates: {candidates_str}")
        if pyin_rate or chord_rate:
            st.caption(f"Cross-validation:{pyin_rate}{chord_rate}")

        st.divider()

        tab_output, tab_table, tab_download = st.tabs(["Output", "Note table", "Download"])

        with tab_output:
            st.code(results["output_text"], language=None)

        with tab_table:
            import pandas as pd
            events       = results["aligned_events"]
            base_cols    = ["note", "start_time", "end_time", "confidence", "lyric"]
            opt_cols     = ["beat_aligned", "pyin_agrees", "chord_fit"]
            display_cols = base_cols + [c for c in opt_cols if events and c in events[0]]
            df = pd.DataFrame(events)[display_cols]
            df["start_time"] = df["start_time"].round(3)
            df["end_time"]   = df["end_time"].round(3)
            df["confidence"] = df["confidence"].round(3)
            st.dataframe(df, use_container_width=True, height=400)

        with tab_download:
            file_ext  = {"table": "txt", "leadsheet": "txt", "csv": "csv",  "json": "json"}
            mime_type = {"table": "text/plain", "leadsheet": "text/plain",
                         "csv": "text/csv", "json": "application/json"}

            st.download_button(
                label=f"⬇  Download as .{file_ext[output_format]}",
                data=results["output_text"],
                file_name=f"transcription.{file_ext[output_format]}",
                mime=mime_type[output_format],
                use_container_width=True,
            )
            st.download_button(
                label="⬇  Download full JSON (all fields)",
                data=json.dumps(results["aligned_events"], indent=2),
                file_name="transcription_full.json",
                mime="application/json",
                use_container_width=True,
            )
