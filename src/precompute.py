# build_utterance_cache_only.py
# Build utterance-level cache parquet once (all cohorts), cutoff-invariant.
# Reuses analyzer (SentenceTransformer loads once).

import os
import hashlib
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

# ---- your project imports ----
from preprocessing import load_text_from_aws_json, clean_text
from segment_utterance import segment_utterances
from normalize_utterances import normalize_utterances
from coherence import count_errors
from mainconcept_normalize import MainConceptAnalyzerNormalize


# -----------------------------
# CONFIG
# -----------------------------
COHORTS = {
    "dementia":  "../../Data/transcripts/bank/dementia/aws/cinderella",
    "aphasia":   "../../Data/transcripts/bank/aphasia/aws/cinderella",
    "controls":  "../../Data/transcripts/bank/controls/aws/cinderella",
}

CACHE_DIR = Path("../data/cache_utts_v2")          # new cache folder (avoid mixing with old schema)
CACHE_INDEX = CACHE_DIR / "_index.parquet"         # combined cache
PARQUET_COMPRESSION = "snappy"


# -----------------------------
# Helpers
# -----------------------------
def participant_id_from_path(json_path: str) -> str:
    base = os.path.splitext(os.path.basename(json_path))[0]
    return base.split("_")[0]

def file_hash_from_normalized_utts(normalized_utts: list[str]) -> str:
    h = hashlib.md5(("||".join(normalized_utts)).encode("utf-8")).hexdigest()
    return h[:12]

def find_json_path(folder_path: str, transcribed_fp: str) -> str:
    """
    Your current code uses: base_transcribed.json exists, but the actual AWS JSON is base.json.
    We'll try base.json first; if missing, fall back to the transcribed file.
    """
    base = os.path.splitext(os.path.basename(transcribed_fp))[0]
    candidate = os.path.abspath(os.path.join(folder_path, f"{base}.json"))
    return candidate if os.path.exists(candidate) else os.path.abspath(transcribed_fp)


# -----------------------------
# Cache builder (cutoff-invariant)
# -----------------------------
def build_one_participant_cache(
    json_path: str,
    cohort: str,
    analyzer: MainConceptAnalyzerNormalize,
    cache_dir: Path = CACHE_DIR,
) -> Path:
    """
    Builds and saves a per-participant utterance cache parquet.

    Caches ONLY cutoff-invariant fields:
      - distance_to_centroid
      - best_idx (argmax concept sim)
      - coherence error_type (per utterance if available)
      - utterance text + normalized version (for debugging)

    Returns path to parquet.
    """
    pid = participant_id_from_path(json_path)

    # Load + segment + normalize
    text = load_text_from_aws_json(json_path)
    cleaned = clean_text(text)
    utts = segment_utterances(cleaned)
    norm_utts = normalize_utterances(utts)

    # Cache filename keyed by participant + hash of normalized utterances
    h = file_hash_from_normalized_utts(norm_utts)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{cohort}__{pid}__{h}.parquet"
    if out_path.exists():
        return out_path

    # ---- MAIN CONCEPT: compute distances + best_idx (NO cutoff logic stored) ----
    analyzer.reset_tracking()
    df_mc = analyzer.get_mainconcept_match(utts, norm_utts, return_score=True)

    # Require cutoff-invariant column best_idx
    if "best_idx" not in df_mc.columns:
        raise RuntimeError(
            "Your get_mainconcept_match() must return 'best_idx' to build a cutoff-invariant cache. "
            "Update MainConceptAnalyzerNormalize.get_mainconcept_match() to include best_idx in the results."
        )

    # ---- COHERENCE: per-utterance error type (optional) ----
    participant_codes = [pid] * len(utts)
    counts, utterance_errors = count_errors(utts, participant_codes)

    if isinstance(utterance_errors, (list, tuple)) and len(utterance_errors) == len(utts):
        error_type_col = list(utterance_errors)
    else:
        error_type_col = [None] * len(utts)

    # NOTE: distance_to_centroid should ideally be full precision in MCA
    # (i.e., stored as float(dist) not round(dist, 4)).
    df_cache = pd.DataFrame({
        "cohort": cohort,
        "label": cohort,  # 3-class label
        "participant": pid,
        "utt_idx": np.arange(len(utts), dtype=int),
        "utterance": utts,
        "normalized_utt": norm_utts,
        "distance_to_centroid": pd.to_numeric(df_mc["distance_to_centroid"], errors="coerce").astype(float),
        "best_idx": pd.to_numeric(df_mc["best_idx"], errors="coerce").astype("Int64"),
        "error_type": pd.Series(error_type_col, dtype="object"),
    })

    df_cache.to_parquet(out_path, index=False, compression=PARQUET_COMPRESSION)
    return out_path


def precompute_all_caches(
    cohorts: dict[str, str] = COHORTS,
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """
    Walk all cohort folders and build per-participant caches.
    Also writes a combined _index.parquet for convenience.
    """
    analyzer = MainConceptAnalyzerNormalize(global_cutoff=0.0)  # cutoff irrelevant here (we're caching distances)
    parquet_paths = []

    for cohort, folder_rel in cohorts.items():
        folder_path = os.path.abspath(folder_rel)
        files = sorted(glob(os.path.join(folder_path, "*_transcribed.json")))
        print(f"[CACHE] cohort={cohort} files={len(files)} folder={folder_path}")

        for fp in files:
            json_path = find_json_path(folder_path, fp)
            pid = participant_id_from_path(json_path)

            try:
                out = build_one_participant_cache(json_path, cohort, analyzer, cache_dir=cache_dir)
                parquet_paths.append(out)
                print(f"  ✅ cached {cohort} {pid} -> {out.name}")
            except Exception as e:
                print(f"  ⚠️ failed {cohort} {pid}: {e}")

    if not parquet_paths:
        raise RuntimeError("No caches were built/read.")

    # Combine into a single index parquet (fast to load later)
    dfs = [pd.read_parquet(p) for p in parquet_paths]
    df_all = pd.concat(dfs, ignore_index=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(CACHE_INDEX, index=False, compression=PARQUET_COMPRESSION)
    print(f"[CACHE] wrote combined cache index: {CACHE_INDEX} rows={len(df_all)}")

    return df_all


def main():
    df_cache = precompute_all_caches()

    # Quick sanity check
    print("\n[INFO] Cached participants:", df_cache["participant"].nunique())
    print("[INFO] Cached utterances:", len(df_cache))
    print("[INFO] Labels:", df_cache["label"].value_counts().to_dict())
    print("[INFO] Columns:", list(df_cache.columns))

    print("\n[INFO] Head:")
    print(df_cache.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
