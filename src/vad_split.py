import os
import glob
import wave
import contextlib
from dataclasses import dataclass
from typing import List, Tuple

import webrtcvad

# ------------ Config ------------
SAMPLE_RATE = 16000
FRAME_MS = 30               # 10, 20, or 30 فقط (webrtcvad requirement)
VAD_MODE = 2                # 0 = least aggressive, 3 = most aggressive
MIN_SPEECH_SEC = 5.0        # ارفض أي ملف بعد القص أقل من 5 ثواني كلام
MAX_SEGMENT_SEC = 30.0      # لو الكلام طويل، هنقطّعه لقطع 30 ثانية
# --------------------------------

@dataclass
class Segment:
    start: float
    end: float

def read_wav_pcm16(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1, "WAV must be mono"
        assert wf.getsampwidth() == 2, "WAV must be 16-bit PCM"
        assert wf.getframerate() == SAMPLE_RATE, f"WAV must be {SAMPLE_RATE}Hz"
        return wf.readframes(wf.getnframes())

def wav_duration_sec(path: str) -> float:
    with contextlib.closing(wave.open(path, "rb")) as wf:
        return wf.getnframes() / float(wf.getframerate())

def frames_from_pcm(pcm: bytes, frame_ms: int) -> List[bytes]:
    frame_bytes = int(SAMPLE_RATE * (frame_ms / 1000.0) * 2)  # 2 bytes per sample
    frames = []
    for i in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        frames.append(pcm[i:i+frame_bytes])
    return frames

def speech_segments(vad: webrtcvad.Vad, frames: List[bytes], frame_ms: int) -> List[Segment]:
    # Simple smoothing: require a short run of speech frames to open a segment, and silence to close it
    speech_flags = [vad.is_speech(f, SAMPLE_RATE) for f in frames]

    # Parameters for hangover
    open_trigger = 5   # frames
    close_trigger = 8  # frames

    segments: List[Segment] = []
    in_speech = False
    start_idx = 0
    speech_run = 0
    silence_run = 0

    for i, is_s in enumerate(speech_flags):
        if not in_speech:
            if is_s:
                speech_run += 1
                if speech_run >= open_trigger:
                    in_speech = True
                    start_idx = i - open_trigger + 1
                    silence_run = 0
            else:
                speech_run = 0
        else:
            if not is_s:
                silence_run += 1
                if silence_run >= close_trigger:
                    end_idx = i - close_trigger + 1
                    segments.append(Segment(start=start_idx * frame_ms / 1000.0,
                                            end=end_idx * frame_ms / 1000.0))
                    in_speech = False
                    speech_run = 0
                    silence_run = 0
            else:
                silence_run = 0

    if in_speech:
        segments.append(Segment(start=start_idx * frame_ms / 1000.0,
                                end=len(frames) * frame_ms / 1000.0))
    return segments

def merge_close_segments(segs: List[Segment], max_gap_sec: float = 0.3) -> List[Segment]:
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: s.start)
    out = [segs[0]]
    for s in segs[1:]:
        last = out[-1]
        if s.start - last.end <= max_gap_sec:
            out[-1] = Segment(start=last.start, end=max(last.end, s.end))
        else:
            out.append(s)
    return out

def write_wav(path: str, pcm: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)

def cut_pcm(pcm: bytes, start_sec: float, end_sec: float) -> bytes:
    start_b = int(start_sec * SAMPLE_RATE) * 2
    end_b = int(end_sec * SAMPLE_RATE) * 2
    return pcm[start_b:end_b]

def chunk_segment(seg: Segment, max_len_sec: float) -> List[Segment]:
    chunks = []
    s = seg.start
    while s < seg.end:
        e = min(seg.end, s + max_len_sec)
        chunks.append(Segment(start=s, end=e))
        s = e
    return chunks

def main(in_dir: str, out_dir: str):
    vad = webrtcvad.Vad(VAD_MODE)

    wavs = sorted(glob.glob(os.path.join(in_dir, "*.wav")))
    if not wavs:
        raise SystemExit(f"No wav files found in {in_dir}")

    kept = 0
    skipped = 0

    for w in wavs:
        pcm = read_wav_pcm16(w)
        dur = wav_duration_sec(w)
        if dur < 1.0:
            skipped += 1
            continue

        frames = frames_from_pcm(pcm, FRAME_MS)
        segs = speech_segments(vad, frames, FRAME_MS)
        segs = merge_close_segments(segs)

        # Flatten segments into PCM, but keep chunking at MAX_SEGMENT_SEC
        base = os.path.splitext(os.path.basename(w))[0]
        out_index = 0
        total_speech = 0.0

        for seg in segs:
            # ignore too-short segs
            if seg.end - seg.start < 0.8:
                continue
            # chunk long segs
            for c in chunk_segment(seg, MAX_SEGMENT_SEC):
                clip = cut_pcm(pcm, c.start, c.end)
                clip_sec = (len(clip) / 2) / SAMPLE_RATE
                if clip_sec < 1.0:
                    continue
                out_path = os.path.join(out_dir, f"{base}_vad_{out_index:03d}.wav")
                write_wav(out_path, clip)
                out_index += 1
                total_speech += clip_sec

        if total_speech >= MIN_SPEECH_SEC:
            kept += 1
        else:
            # Remove tiny outputs (quality gate)
            for f in glob.glob(os.path.join(out_dir, f"{base}_vad_*.wav")):
                try:
                    os.remove(f)
                except OSError:
                    pass
            skipped += 1

    print(f"Done. kept_files={kept}, skipped_files={skipped}")
    print(f"Output: {out_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python src/vad_split.py <in_dir_wav16k> <out_dir>")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])

