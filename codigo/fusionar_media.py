
import pandas as pd
import numpy as np
import glob, os, re
from pathlib import Path

DATA_DIR = "./diarizado/media"  # carpeta con los CSV (p.ej. "/ruta/a/tus/csvs")
OUT_CSV_MERGED  = "./diarizado/media_v2"
MAX_GAP_SEC = 0.75  # segundos máximo entre turnos para fusionar
MAX_DURATION_SEC = None  # duración máxima de un turno (None = sin límite)

def parse_audio_id_from_filename(path: str) -> str:
    """Extrae ID de audio del nombre del fichero:
    - Si hay texto entre corchetes [ID], devuelve eso
    - Si no, usa el nombre (sin extensión) 'normalizado'
    """
    base = os.path.basename(path)
    m = re.search(r"\[([^\]]+)\]", base)
    if m:
        return m.group(1)
    stem = os.path.splitext(base)[0]
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9_]+", "", stem)
    return stem[:50]

def try_read_csv(path: str) -> pd.DataFrame:
    """Intenta leer con separador ';' y ',' y varias codificaciones."""
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    seps = [";", ","]
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
                # si todo quedó en 1 columna, probablemente el separador no era ese
                if df.shape[1] == 1 and sep == ",":
                    continue
                return df
            except Exception as e:
                last_err = e
    raise RuntimeError(f"No se pudo leer {path}: {last_err}")

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza a columnas estándar: start_sec, end_sec, speaker, text (solo UNA de speaker)."""
    df = df.copy()

    # Mapeo de nombres posibles -> estándar
    colmap = {
        # tiempos
        "start_sec": "start_sec",
        "end_sec": "end_sec",
        # hablante (¡única columna!)
        "speaker": "speaker",
        # texto
        "text": "text",
    }

    # Renombrado según mapeo
    ren = {}
    for c in df.columns:
        low = c.strip().lower()
        if low in colmap:
            ren[c] = colmap[low]
    if ren:
        df = df.rename(columns=ren)

    # Tipos + limpieza
    for tcol in ("start_sec","end_sec"):
        if tcol in df.columns:
            df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
    if "text" in df.columns:
        df["text"] = df["text"].astype(str).replace({"nan": ""})

    # Quitar filas sin texto
    if "text" in df.columns:
        df = df[df["text"].astype(str).str.strip() != ""]

    # Mantener SOLO las columnas que nos interesan si existen
    cols = [c for c in ["start_sec","end_sec","speaker","text"] if c in df.columns]
    return df[cols].copy()

def add_computed_fields(df: pd.DataFrame, audio_id: str) -> pd.DataFrame:
    """Añade audio_id, duration_sec, n_chars, n_words y ordena columnas."""
    df = df.copy()
    df["audio_id"] = audio_id

    # duration
    if {"start_sec","end_sec"}.issubset(df.columns):
        df["duration_sec"] = (df["end_sec"] - df["start_sec"]).astype(float)
    else:
        df["duration_sec"] = np.nan

    # métricas rápidas
    if "text" in df.columns:
        s = df["text"].astype(str)
        df["n_chars"] = s.str.len()                # nº de caracteres (incluye espacios y signos)
        df["n_words"] = s.str.split().apply(len)  # nº de tokens básicos (split por espacios)
    else:
        df["text"] = ""
        df["n_chars"] = 0
        df["n_words"] = 0

    # Orden final
    ordered = ["audio_id","start_sec","end_sec","duration_sec","speaker","text","n_chars","n_words"]
    final_cols = [c for c in ordered if c in df.columns] + [c for c in df.columns if c not in ordered]
    return df[final_cols]

def fuse_consecutive_turns(df: pd.DataFrame) -> pd.DataFrame:
    """Fusiona segmentos consecutivos del mismo speaker por audio_id, con reglas de hueco y duración."""
    if df.empty:
        return df.copy()

    merged_rows = []
    for audio_id, g in df.groupby("audio_id", sort=False):
        g = g.sort_values(["start_sec","end_sec"], kind="mergesort").reset_index(drop=True)
        cur = g.iloc[0].to_dict()
        cur_spk = str(cur.get("speaker"))

        merged_count = 1
        for i in range(1, len(g)):
            row = g.iloc[i].to_dict()
            row_spk = str(row.get("speaker"))

            # Hueco entre turnos
            gap = (row.get("start_sec", np.nan) - cur.get("end_sec", np.nan))
            gap_ok = (pd.isna(gap) or gap <= MAX_GAP_SEC)

            # Duración si se fusiona
            new_end = np.nanmax([cur.get("end_sec", np.nan), row.get("end_sec", np.nan)])
            new_start = cur.get("start_sec", np.nan)
            new_duration = (new_end - new_start) if (pd.notna(new_end) and pd.notna(new_start)) else np.nan
            duration_ok = True if (MAX_DURATION_SEC is None or pd.isna(new_duration)) else (new_duration <= MAX_DURATION_SEC)

            if row_spk == cur_spk and gap_ok and duration_ok:
                # Fusionar
                cur["end_sec"] = new_end
                cur["duration_sec"] = new_duration if pd.notna(new_duration) else cur.get("duration_sec", np.nan)
                # Concatena texto con un espacio
                cur["text"] = (str(cur.get("text","")).strip() + " " + str(row.get("text","")).strip()).strip()
                merged_count += 1
            else:
                # Finaliza el acumulado actual
                cur["n_chars"] = len(str(cur.get("text","")))
                cur["n_words"] = len(str(cur.get("text","")).split())
                merged_rows.append(cur)

                # Nuevo acumulado
                cur = row
                cur_spk = row_spk
                merged_count = 1

        # Último acumulado
        cur["n_chars"] = len(str(cur.get("text","")))
        cur["n_words"] = len(str(cur.get("text","")).split())
        merged_rows.append(cur)

    merged_df = pd.DataFrame(merged_rows)
    # Orden recomendable (ahora incluye merged_segments)
    ordered = ["audio_id","start_sec","end_sec","duration_sec","speaker","text","n_chars","n_words"]
    ordered = [c for c in ordered if c in merged_df.columns] + [c for c in merged_df.columns if c not in ordered]
    return merged_df[ordered]

def main():
    csv_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not csv_paths:
        raise SystemExit("No se encontraron CSVs en DATA_DIR.")

    for csv_path in csv_paths:
        audio_id = parse_audio_id_from_filename(csv_path)
        raw = try_read_csv(csv_path)
        std = standardize_columns(raw)
        std = add_computed_fields(std, audio_id)
        merged = fuse_consecutive_turns(std)
        Path(OUT_CSV_MERGED).mkdir(parents=True, exist_ok=True)
        merged.to_csv(f"{OUT_CSV_MERGED}/{Path(csv_path).stem}.csv", index=False, encoding="utf-8")
        print(f"Guardado: {OUT_CSV_MERGED}/{Path(csv_path).stem}.csv -> {len(merged)} filas")

if __name__ == "__main__":
    main()