#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import glob
import torch
import datetime
from pathlib import Path
import whisper
from rich import print

def hhmmss_msec(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    td = datetime.timedelta(seconds=float(seconds))
    # td -> H:MM:SS.mmmmmm  ; convert to SRT "HH:MM:SS,mmm"
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(round((float(seconds) - int(seconds)) * 1000.0))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def write_srt(segments, srt_path: Path):
    with srt_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = hhmmss_msec(seg["start"])
            end = hhmmss_msec(seg["end"])
            text = (seg.get("text") or "").strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def main():

    in_dir = Path("./audios/media")  # separados pizarra y media inglesa
    out_tx = Path("./transcripts/media")
    out_tx.mkdir(parents=True, exist_ok=True)
    model_size = "medium"
    language = "es"

    audio_paths = sorted(glob.glob(str(in_dir / "*.mp3")))
    if not audio_paths:
        print("[bold red]No se encontraron MP3 en ./[/bold red]")
        return

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = device == "cuda"

    print(f"[bold]Usando dispositivo:[/bold] {device}  |  Modelo Whisper: {model_size}  |  fp16={fp16}")

    # Cargar el modelo UNA vez
    model = whisper.load_model(model_size, device=device)

    for ap in audio_paths:
        ap = Path(ap)
        stem = ap.stem
        print(f"\n[cyan]Transcribiendo[/cyan]: {ap.name}")
        if (out_tx / f"{stem}.segments.json").exists():
            print(f"[yellow]Aviso:[/yellow] ya existe {stem}.segments.json, se omite.")
            continue
        # Transcribir
        result = model.transcribe(str(ap), language=language, fp16=fp16)

        # Guardar JSON con segmentos
        json_path = out_tx / f"{stem}.segments.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Guardar TXT (texto plano)
        txt_path = out_tx / f"{stem}.txt"
        full_text = (result.get("text") or "").strip()
        txt_path.write_text(full_text, encoding="utf-8")

        # Guardar SRT
        srt_path = out_tx / f"{stem}.srt"
        write_srt(result.get("segments", []), srt_path)

        print(f"[green]OK[/green] -> {json_path.name}, {txt_path.name}, {srt_path.name}")

if __name__ == "__main__":
    main()
