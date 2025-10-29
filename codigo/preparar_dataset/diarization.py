import os
import json
import argparse
import glob
from pathlib import Path
from rich import print
import torch
from pyannote.audio import Pipeline
from typing import List, Dict, Any
import datetime
import csv
import torchaudio
import subprocess
import tempfile

def hhmmss_msec(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    td = datetime.timedelta(seconds=float(seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(round((float(seconds) - int(seconds)) * 1000.0))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def load_transcript_segments(transcripts_dir: Path, stem: str) -> List[Dict[str, Any]]:
    json_path = transcripts_dir / f"{stem}.segments.json"
    if not json_path.exists():
        return []
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("segments", [])

def intervals_overlap(a_start, a_end, b_start, b_end) -> float:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)

def best_speaker_for_segment(seg, spk_segments):
    s_start, s_end = float(seg["start"]), float(seg["end"])
    best_label, best_overlap = None, 0.0
    for spk_seg in spk_segments:
        ov = intervals_overlap(s_start, s_end, spk_seg["start"], spk_seg["end"])
        if ov > best_overlap:
            best_overlap = ov
            best_label = spk_seg["speaker"]
    return best_label

def write_rttm(spk_segments, path: Path, uri: str):
    with path.open("w", encoding="utf-8") as f:
        for s in spk_segments:
            start = s["start"]
            dur = s["end"] - s["start"]
            label = s["speaker"]
            f.write(f"SPEAKER {uri} 1 {start:.3f} {dur:.3f} <NA> <NA> {label} <NA>\n")

def write_srt(diarized_segments, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(diarized_segments, start=1):
            start = hhmmss_msec(seg["start"])
            end = hhmmss_msec(seg["end"])
            spk = seg["speaker"]
            text = seg.get("text", "").strip()
            line = f"{spk}: {text}" if text else f"{spk}"
            f.write(f"{i}\n{start} --> {end}\n{line}\n\n")

def write_csv(diarized_segments, path: Path):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["start_sec", "end_sec", "speaker", "text"])
        for seg in diarized_segments:
            writer.writerow([f"{seg['start']:.3f}", f"{seg['end']:.3f}", seg["speaker"], seg.get("text","")])

def main():

    in_dir = Path("./audios/pizarra")
    out_diar = Path("./diarizado_v3/pizarra")
    out_diar.mkdir(parents=True, exist_ok=True)
    n_hablantes = 4

    transcripts_dir = Path("./transcripts/pizarra")
    if not transcripts_dir.exists():
        print("[yellow]Aviso: no se encontró carpeta de transcripciones; se generará diarización sin texto.[/yellow]")

    audio_paths = sorted(glob.glob(str(in_dir / "*.mp3")))
    if not audio_paths:
        print("[bold red]No se encontraron MP3 en la carpeta de entrada.[/bold red]")
        return

    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("[bold red]ERROR:[/bold red] Define HUGGINGFACE_TOKEN o rellena el token en el script.")
        return

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bold]Usando dispositivo:[/bold] {device}")

    print("[cyan]Cargando pipeline pyannote/speaker-diarization-community-1...[/cyan]")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", use_auth_token=token)

    if device == "cuda":
        pipeline.to(torch.device("cuda"))

    for ap in audio_paths:
        ap = Path(ap)
        stem = ap.stem
        print(f"\n[cyan]Diarizando[/cyan]: {ap.name}")
        if (out_diar / f"{stem}.diarized.json").exists():
            print(f"[yellow]Aviso:[/yellow] ya existe {stem}.diarized.json, se omite.")
            continue
        try:
            # Cargar MP3 en memoria y pasar al pipeline
            waveform, sr = torchaudio.load(str(ap))
            result = pipeline({"waveform": waveform, "sample_rate": sr},
                            num_speakers=int(n_hablantes))
        except Exception as e:
            # Si la decodificación a chunks falla, reconvertimos a WAV 16k mono y reintentamos
            print(f"[yellow]Aviso:[/yellow] {e}\n[cyan]Reintentando vía WAV 16k temporal...[/cyan]")
            with tempfile.TemporaryDirectory() as td:
                wav_tmp = Path(td) / f"{ap.stem}_16k.wav"
                cmd = [
                    "ffmpeg", "-nostdin", "-y",
                    "-i", str(ap),
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                    str(wav_tmp)
                ]
                subprocess.run(cmd, check=True)
                result = pipeline(str(wav_tmp), num_speakers=int(n_hablantes))

        annotation = getattr(result, "speaker_diarization", result)

        # Convertir a lista simple de segmentos de speaker
        spk_segments = []
        try:
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                if isinstance(speaker, (int, float)):
                    spk_label = f"SPEAKER_{int(speaker):02d}"
                else:
                    spk_label = str(speaker)
                spk_segments.append({
                    "speaker": spk_label,
                    "start": float(turn.start),
                    "end": float(turn.end),
                })
        except Exception:
            try:
                for turn, speaker in annotation:
                    if isinstance(speaker, (int, float)):
                        spk_label = f"SPEAKER_{int(speaker):02d}"
                    else:
                        spk_label = str(speaker)
                    spk_segments.append({
                        "speaker": spk_label,
                        "start": float(turn.start),
                        "end": float(turn.end),
                    })
            except Exception as e:
                print(f"[bold red]No se pudo interpretar la salida del pipeline:[/bold red] {e}")
                continue

        # Ordenar por tiempo
        spk_segments.sort(key=lambda x: (x["start"], x["end"]))

        # Guardar RTTM
        rttm_path = out_diar / f"{stem}.rttm"
        write_rttm(spk_segments, rttm_path, uri=stem)

        # Fusionar con transcripción
        tx_segments = load_transcript_segments(transcripts_dir, stem)
        diarized_segments = []
        if tx_segments:
            for seg in tx_segments:
                assigned = best_speaker_for_segment(seg, spk_segments) or "SPEAKER_??"
                diarized_segments.append({
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "speaker": assigned,
                    "text": (seg.get("text") or "").strip()
                })
        else:
            diarized_segments = [
                {"start": s["start"], "end": s["end"], "speaker": s["speaker"], "text": ""}
                for s in spk_segments
            ]

        # Guardar JSON
        diar_json_path = out_diar / f"{stem}.diarized.json"
        with diar_json_path.open("w", encoding="utf-8") as f:
            json.dump({"segments": diarized_segments}, f, ensure_ascii=False, indent=2)

        # SRT con speaker + texto
        diar_srt_path = out_diar / f"{stem}.diarized.srt"
        write_srt(diarized_segments, diar_srt_path)

        # CSV
        diar_csv_path = out_diar / f"{stem}.diarized.csv"
        write_csv(diarized_segments, diar_csv_path)

        print(f"[green]OK[/green] -> {rttm_path.name}, {diar_json_path.name}, {diar_srt_path.name}, {diar_csv_path.name}")

if __name__ == "__main__":
    main()
