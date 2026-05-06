"""
meloScribe — Streamlit UI
Run with: streamlit run app.py
"""

import sys
import os
import tempfile
import time
from pathlib import Path

import streamlit as st

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="meloScribe",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom CSS ────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Tighten up the sidebar */
    section[data-testid="stSidebar"] { min-width: 300px; max-width: 320px; }
    /* Phase status pills */
    .phase-ok   { color: #2ecc71; font-weight: 600; }
    .phase-warn { color: #f39c12; font-weight: 600; }
    .phase-err  { color: #e74c3c; font-weight: 600; }
    /* Note table tweaks */
    .stDataFrame { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Pipeline import (lazy, with friendly error) ───────────────────────────────

@st.cache_resource
def load_pipeline():
    try:
        from stemmer import isolate_vocals
        from pitch_detector import detect_pitch
        from note_mapper import map_notes, detect_key
        from lyric_aligner import align_lyrics_to_notes
        from output import format_output
        from beat_tracker import get_beat_grid, BeatTracker
        from config import load_config
        return dict(
            isolate_vocals=isolate_vocals,
            detect_pitch=detect_pitch,
            map_notes=map_notes,
            detect_key=detect_key,
            align_lyrics_to_notes=align_lyrics_to_notes,
            format_output=format_output,
            get_beat_grid=get_beat_grid,
            BeatTracker=BeatTracker,
            load_config=load_config,
        )
    except ImportError as e:
        return {"error": str(e)}

pipeline = load_pipeline()

# ── Sidebar — settings ────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎵 meloScribe")
    st.caption("Melody transcription for saxophone")
    st.divider()

    st.subheader("Settings")

    # Load config defaults if available
    cfg = {}
    if "error" not in pipeline:
        try:
            cfg = pipeline["load_config"]()
        except Exception:
            pass

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.0, max_value=1.0,
        value=float(cfg.get("confidence", 0.85)),
        step=0.05,
        help="Minimum pitch detection confidence. Higher = fewer but more accurate notes."
    )

    transpose = st.number_input(
        "Transpose (semitones)",
        min_value=-24, max_value=24,
        value=int(cfg.get("transpose", 9)),
        step=1,
        help="9 = alto sax (Eb). 0 = concert pitch."
    )

    output_format = st.selectbox(
        "Output format",
        options=["table", "leadsheet", "csv", "json"],
        index=["table", "leadsheet", "csv", "json"].index(cfg.get("format", "table")),
    )

    st.divider()
    st.subheader("Optional enhancements")

    use_dual_pitch = st.toggle(
        "PYIN cross-validation",
        value=bool(cfg.get("dual_pitch", False)),
        help="Runs a second pitch model for better accuracy. Adds ~15s."
    )

    use_chord_context = st.toggle(
        "Chord context",
        value=bool(cfg.get("chord_context", False)),
        help="Extracts chord timeline from the backing track. Requires full MP3 input."
    )

    st.divider()
    st.caption("meloScribe — passion project 🎷")

# ── Main area ─────────────────────────────────────────────────────────────────

st.header("Melody Transcription")

if "error" in pipeline:
    st.error(f"Pipeline failed to load: {pipeline['error']}\n\nMake sure all pipeline modules are installed.")
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
        audio_file = st.file_uploader(
            "Upload MP3", type=["mp3"],
            help="The full song — Demucs will separate the vocals automatically."
        )
        vocals_file = None
    else:
        audio_file = None
        vocals_file = st.file_uploader(
            "Upload vocals.wav", type=["wav"],
            help="Already-stemmed vocals track — skips the slow Demucs step."
        )

with col_right:
    st.subheader("Lyrics (optional)")
    lyrics_file = st.file_uploader(
        "Upload .lrc file", type=["lrc"],
        help="Time-synced lyrics. If omitted, notes will have no lyric attached."
    )
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

    # Write uploaded files to temp dir
    tmp_dir = tempfile.mkdtemp()
    results = {}

    try:
        # Save audio
        if audio_file:
            audio_path = os.path.join(tmp_dir, audio_file.name)
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())
            input_path  = audio_path
            vocals_path = None
        else:
            vocals_path_tmp = os.path.join(tmp_dir, vocals_file.name)
            with open(vocals_path_tmp, "wb") as f:
                f.write(vocals_file.getbuffer())
            input_path  = None
            vocals_path = vocals_path_tmp

        # Save lyrics
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
                st.write("⏳ This takes 1–3 minutes depending on your hardware.")
                t0 = time.time()
                vocals_path = pipeline["isolate_vocals"](input_path)
                elapsed = time.time() - t0
                st.write(f"✓ Vocals isolated in {elapsed:.0f}s → `{Path(vocals_path).name}`")
                s1.update(label=f"Phase 1 — Vocals separated ({elapsed:.0f}s) ✓", state="complete")

        # ── Phase 2: Pitch ───────────────────────────────────────────────────
        with st.status("Phase 2 — Detecting pitch…", expanded=True) as s2:
            t0 = time.time()
            pitch_data = pipeline["detect_pitch"](vocals_path)
            elapsed = time.time() - t0
            st.write(f"✓ {len(pitch_data):,} pitch frames detected in {elapsed:.1f}s")
            s2.update(label=f"Phase 2 — Pitch detected ({len(pitch_data):,} frames) ✓", state="complete")

        # ── Phase 2b: Beat grid ──────────────────────────────────────────────
        beat_grid = None
        with st.status("Phase 2b — Beat grid…", expanded=False) as s2b:
            try:
                beat_grid = pipeline["get_beat_grid"](vocals_path)
                st.write(f"✓ {beat_grid['bpm']:.1f} BPM, {len(beat_grid['beat_times'])} beats")
                s2b.update(label=f"Phase 2b — Beat grid: {beat_grid['bpm']:.1f} BPM ✓", state="complete")
            except Exception as e:
                st.write(f"⚠ Beat detection failed: {e}")
                s2b.update(label="Phase 2b — Beat grid skipped ⚠", state="error")

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
                    st.write(f"⚠ PYIN failed: {e}")
                    s2c.update(label="Phase 2c — PYIN skipped ⚠", state="error")

        # ── Phase 2d: Chord context ──────────────────────────────────────────
        chord_timeline = None
        if use_chord_context:
            with st.status("Phase 2d — Chord context…", expanded=True) as s2d:
                try:
                    from chord_tracker import get_chord_timeline
                    from stemmer import get_stem_paths
                    stem_paths = get_stem_paths(input_path or vocals_path)
                    beat_times = beat_grid["beat_times"] if beat_grid else []
                    chord_timeline = get_chord_timeline(stem_paths["other"], beat_times)
                    st.write(f"✓ {len(chord_timeline)} chord entries")
                    s2d.update(label=f"Phase 2d — Chords: {len(chord_timeline)} entries ✓", state="complete")
                except Exception as e:
                    st.write(f"⚠ Chord extraction failed: {e}")
                    s2d.update(label="Phase 2d — Chord context skipped ⚠", state="error")

        # ── Phase 3: Note mapping ────────────────────────────────────────────
        with st.status("Phase 3 — Mapping notes…", expanded=False) as s3:
            note_events = pipeline["map_notes"](
                pitch_data, confidence,
                transpose=transpose,
                pyin_data=pyin_data,
                chord_timeline=chord_timeline,
            )
            if beat_grid:
                tracker = pipeline["BeatTracker"]()
                note_events = tracker.annotate_beat_alignment(note_events, beat_grid)

            pyin_rate = ""
            if pyin_data:
                agreed = sum(1 for e in note_events if e.get("pyin_agrees"))
                pyin_rate = f" · PYIN: {agreed}/{len(note_events)} agreed"

            chord_rate = ""
            if chord_timeline:
                fit = sum(1 for e in note_events if e.get("chord_fit"))
                chord_rate = f" · Chord fit: {fit}/{len(note_events)}"

            s3.update(
                label=f"Phase 3 — {len(note_events)} note events{pyin_rate}{chord_rate} ✓",
                state="complete"
            )

        # ── Phase 4: Lyrics ──────────────────────────────────────────────────
        with st.status("Phase 4 — Aligning lyrics…", expanded=False) as s4:
            aligned_events = pipeline["align_lyrics_to_notes"](note_events, lyrics_path)
            matched = sum(1 for e in aligned_events if e.get("lyric") is not None)
            if lyrics_path:
                s4.update(label=f"Phase 4 — Lyrics: {matched}/{len(aligned_events)} matched ✓", state="complete")
            else:
                s4.update(label="Phase 4 — No lyrics provided", state="complete")

        # ── Key detection ────────────────────────────────────────────────────
        key_result = pipeline["detect_key"](aligned_events)

        # ── Phase 5: Format output ───────────────────────────────────────────
        output_text = pipeline["format_output"](aligned_events, format_type=output_format)

        results = {
            "aligned_events": aligned_events,
            "key_result": key_result,
            "output_text": output_text,
            "note_count": len(aligned_events),
            "pyin_rate": pyin_rate,
            "chord_rate": chord_rate,
            "bpm": beat_grid["bpm"] if beat_grid else None,
        }

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        import traceback
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())
        results = {}

    # ── Results ───────────────────────────────────────────────────────────────

    if results:
        st.divider()
        st.subheader("Results")

        # Summary metrics
        key = results["key_result"]
        candidates_str = "  ·  ".join(
            f"{k} ({v:.0%})" for k, v in key["candidates"]
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Notes detected", results["note_count"])
        m2.metric("Estimated key", key["key"])
        m3.metric("Key confidence", f"{key['score']:.0%}")
        if results["bpm"]:
            m4.metric("Tempo", f"{results['bpm']:.0f} BPM")
        else:
            m4.metric("Tempo", "—")

        st.caption(f"Key candidates: {candidates_str}")

        st.divider()

        # Tabs: formatted output + raw table
        tab_output, tab_table, tab_download = st.tabs(["Output", "Note table", "Download"])

        with tab_output:
            st.code(results["output_text"], language=None)

        with tab_table:
            import pandas as pd
            # Show clean subset of columns
            events = results["aligned_events"]
            display_cols = ["note", "start_time", "end_time", "confidence", "lyric"]
            optional_cols = ["beat_aligned", "pyin_agrees", "chord_fit"]
            for col in optional_cols:
                if events and col in events[0]:
                    display_cols.append(col)

            df = pd.DataFrame(events)[display_cols]
            df["start_time"] = df["start_time"].round(3)
            df["end_time"]   = df["end_time"].round(3)
            df["confidence"] = df["confidence"].round(3)
            st.dataframe(df, use_container_width=True, height=400)

        with tab_download:
            st.write("Download the transcription in your chosen format:")

            file_ext = {"table": "txt", "leadsheet": "txt", "csv": "csv", "json": "json"}
            mime_type = {"table": "text/plain", "leadsheet": "text/plain",
                         "csv": "text/csv", "json": "application/json"}

            st.download_button(
                label=f"⬇  Download as .{file_ext[output_format]}",
                data=results["output_text"],
                file_name=f"transcription.{file_ext[output_format]}",
                mime=mime_type[output_format],
                use_container_width=True,
            )

            # Also always offer JSON of full events
            import json
            json_full = json.dumps(results["aligned_events"], indent=2)
            st.download_button(
                label="⬇  Download full JSON (all fields)",
                data=json_full,
                file_name="transcription_full.json",
                mime="application/json",
                use_container_width=True,
            )
