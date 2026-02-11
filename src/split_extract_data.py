from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sentence_transformers import util
import torch
from mainconcept_normalize import MainConceptAnalyzerNormalize
from coherence import (
    count_errors_fold_one_participant,
    calculate_coherence_error_percentages,
)

CACHE_INDEX = Path("../data/cache_utts_v2/_index.parquet")
SAVE_DIR = Path("../data/classification/fold_features_traincutoff_designB")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH = Path("../data/cache_utts_v2/dementia__06-1__4082a44679b8.parquet")

N_SPLITS = 5
SEED = 42
CUTOFF_K = 1.0
SIM_THR = 0.8

ERROR_TYPES = [
    "Topic Switching",
    "Missing Referent",
    "Tangential Utterance",
    "Propositional Repetition",
    "Filler",
    "Conceptual Incongruence",
    "Coherent",
]

def fit_cutoff_mean_plus_kstd(distances: np.ndarray, k: float = 1.0) -> float:
    distances = distances[np.isfinite(distances)]
    if len(distances) == 0:
        raise ValueError("No finite distances to fit cutoff from.")
    mu = float(distances.mean())
    sd = float(distances.std(ddof=0))
    return mu + k * sd

def order_difference_ratio(seq):
    n = len(seq)
    if n < 2:
        return 0.0
    violations = 0
    for i in range(n):
        for j in range(i+1, n):
            if seq[i] > seq[j]:
                violations += 1
    return violations / (n * (n - 1) / 2)

def compute_repeated_flags(
    analyzer: MainConceptAnalyzerNormalize,
    normalized_utts: list[str],
    is_match: list[bool],
    sim_thr: float = 0.8,
) -> list[bool]:
    """
    IDENTICAL to old MainConceptAnalyzerNormalize.is_repeated_utt behavior,
    when called inside `if is_match:`.

    Old behavior:
      - only check repetition for matched utterances
      - repeated if max cosine sim to ANY previous stored utterance > sim_thr
      - if repeated: return True and DO NOT add this utterance to history
      - if not repeated: add to history
    """
    rep = [False] * len(normalized_utts)
    prev_embs = []  # list[tensor]

    # batch encode for speed (still same model)
    embs = analyzer.embedder.encode(
        normalized_utts,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    for i, (emb, m) in enumerate(zip(embs, is_match)):
        if not m:
            rep[i] = False
            continue

        if prev_embs:
            prev_mat = torch.stack(prev_embs)   # shape: (n_prev, dim)
            sims = util.cos_sim(emb.unsqueeze(0), prev_mat)
            max_sim = float(sims.max().item())
            if max_sim > sim_thr:
                rep[i] = True
                # IMPORTANT: do NOT append emb (matches old code)
                continue

        # not repeated → store
        prev_embs.append(emb)
        rep[i] = False

    return rep

def compute_topic_switch_flags(
    analyzer: MainConceptAnalyzerNormalize,
    utterances: list[str],
) -> list[bool]:
    """
    Matches old pipeline: analyzer.is_topic_switching(prev, curr)
    """
    flags = [False] * len(utterances)
    for i in range(1, len(utterances)):
        flags[i] = bool(analyzer.is_topic_switching(utterances[i-1], utterances[i]))
    return flags

def build_participant_row(dfp: pd.DataFrame, cutoff: float, analyzer: MainConceptAnalyzerNormalize) -> dict:
    dfp = dfp.sort_values("utt_idx").reset_index(drop=True)
    utts = dfp["utterance"].astype(str).tolist()
    norm_utts = dfp["normalized_utt"].astype(str).tolist()
    dists = dfp["distance_to_centroid"].astype(float).tolist()
    best_idx = dfp["best_idx"].astype(int).tolist()

    # fold match based on cutoff
    is_match = [dist <= cutoff for dist in dists]

    # matched concept only if matched
    concepts = analyzer.concepts
    matched_concepts = [concepts[i] if m else None for i, m in zip(best_idx, is_match)]

    # repeated only among matched (mirrors your coherence logic)
    repeated_flags = compute_repeated_flags(
        analyzer=analyzer,
        normalized_utts=norm_utts,
        is_match=is_match,
        sim_thr=0.8,   # EXACT old threshold
    )

    # topic switching within participant
    topic_switch_flags = compute_topic_switch_flags(analyzer, utts)

    # fold coherence counts (same shape as count_errors())
    counts, _ = count_errors_fold_one_participant(
        utterances=utts,
        is_mainconcept_match=is_match,
        matched_concepts=matched_concepts,
        is_repeated_flags=repeated_flags,
        topic_switch_flags=topic_switch_flags,
    )

    total_utterances = len(utts)

    # participant-level MCA features (match your main.py outputs)
    num_total_mc = int(sum(is_match))
    num_unique_mc = int(pd.Series([cid for cid, m in zip(best_idx, is_match) if m]).nunique())
    seq = [cid for cid, m in zip(best_idx, is_match) if m]
    seq_score = float(order_difference_ratio(seq))

    # local/global as you do
    local_pct, global_pct = calculate_coherence_error_percentages(counts, total_utterances)

    row = {
        "participant": dfp["participant"].iloc[0],
        "label": dfp["label"].iloc[0],
        "total_utterances": total_utterances,
        "distance_to_centroid": float(np.mean(dists)) if total_utterances else np.nan,
        "num_unique_main_concepts": num_unique_mc,
        "num_total_main_concepts": num_total_mc,
        "unique_main_concept_match_ratio": num_unique_mc / 34.0,
        "total_main_concept_match_ratio": num_total_mc / max(total_utterances, 1),
        "sequence_score_mainconcept": seq_score,
        "local_coherence_mainconcept": local_pct,
        "global_coherence_mainconcept": global_pct,
    }

    # per error type counts/ratios exactly like your loop
    for error_type in ERROR_TYPES:
        num_col = f'num_{error_type.lower().replace(" ", "_")}'
        ratio_col = f'ratio_{error_type.lower().replace(" ", "_")}_percent'
        num = counts.get(error_type, 0)
        ratio = (num / total_utterances * 100.0) if total_utterances > 0 else 0.0
        row[num_col] = int(num)
        row[ratio_col] = round(ratio, 2)

    return row

def extract_features_fold(df_cache: pd.DataFrame, pids: set[str], cutoff: float, analyzer: MainConceptAnalyzerNormalize) -> pd.DataFrame:
    df_split = df_cache[df_cache["participant"].isin(pids)].copy()
    rows = []
    for pid, dfp in df_split.groupby("participant", sort=False):
        rows.append(build_participant_row(dfp, cutoff, analyzer))
    return pd.DataFrame(rows)

def run_folds_save_features(df_cache: pd.DataFrame):
    df_part = df_cache[["participant", "label"]].drop_duplicates("participant").reset_index(drop=True)
    label_map = {name: i for i, name in enumerate(sorted(df_part["label"].unique()))}
    df_part["y"] = df_part["label"].map(label_map).astype(int)

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    analyzer = MainConceptAnalyzerNormalize(global_cutoff=0.0)  # reuse model load
    summary = []

    for fold, (tr, te) in enumerate(sgkf.split(np.zeros(len(df_part)), df_part["y"], df_part["participant"])):
        train_pids = set(df_part.loc[tr, "participant"])
        test_pids  = set(df_part.loc[te, "participant"])

        df_train_utts = df_cache[df_cache["participant"].isin(train_pids)]
        cutoff = fit_cutoff_mean_plus_kstd(df_train_utts["distance_to_centroid"].to_numpy(), k=CUTOFF_K)

        X_train = extract_features_fold(df_cache, train_pids, cutoff, analyzer)
        X_test  = extract_features_fold(df_cache, test_pids, cutoff, analyzer)

        for out, split in [(X_train, "train"), (X_test, "test")]:
            out.insert(0, "fold", fold)
            out.insert(1, "split", split)
            out.insert(2, "cutoff_from", "train")
            out.insert(3, "cutoff", cutoff)

        train_path = SAVE_DIR / f"fold{fold:02d}_train_features_traincutoff_k{CUTOFF_K}.csv"
        test_path  = SAVE_DIR / f"fold{fold:02d}_test_features_traincutoff_k{CUTOFF_K}.csv"
        X_train.to_csv(train_path, index=False)
        X_test.to_csv(test_path, index=False)

        print(f"[FOLD {fold}] cutoff(train)={cutoff:.6f}\n  {train_path}\n  {test_path}")

        summary.append({
            "fold": fold,
            "cutoff_from": "train",
            "cutoff": cutoff,
            "n_train_participants": len(train_pids),
            "n_test_participants": len(test_pids),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        })

    summary_df = pd.DataFrame(summary)
    summary_path = SAVE_DIR / f"fold_summary_traincutoff_k{CUTOFF_K}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[DONE] {summary_path}")

def main():
    if not CACHE_INDEX.exists():
        raise FileNotFoundError(f"Cache index not found: {CACHE_INDEX}")

    df_cache = pd.read_parquet(CACHE_INDEX)

    needed = ["participant", "label", "utt_idx", "utterance", "distance_to_centroid", "best_idx"]
    missing = [c for c in needed if c not in df_cache.columns]
    if missing:
        raise KeyError(f"CACHE missing columns: {missing}. got={list(df_cache.columns)}")

    df_cache = df_cache.dropna(subset=["distance_to_centroid", "best_idx"]).reset_index(drop=True)
    df_cache["utt_idx"] = pd.to_numeric(df_cache["utt_idx"], errors="coerce")
    df_cache["best_idx"] = pd.to_numeric(df_cache["best_idx"], errors="coerce").astype("Int64")
    df_cache = df_cache.dropna(subset=["utt_idx", "best_idx"]).copy()

    print("[INFO] participants:", df_cache["participant"].nunique())
    print("[INFO] utterances:", len(df_cache))
    print("[INFO] labels:", df_cache["label"].value_counts().to_dict())

    run_folds_save_features(df_cache)

if __name__ == "__main__":
    main()

# =========================
# MAIN
# # =========================
# def main():

#     if not PARQUET_PATH.exists():
#         raise FileNotFoundError(PARQUET_PATH)

#     df = pd.read_parquet(PARQUET_PATH)

#     needed = [
#         "participant",
#         "label",
#         "utt_idx",
#         "utterance",
#         "normalized_utt",
#         "distance_to_centroid",
#         "best_idx",
#     ]

#     for c in needed:
#         if c not in df.columns:
#             raise ValueError(f"Missing column: {c}")

#     df = df.sort_values("utt_idx").reset_index(drop=True)

#     utts = df["utterance"].astype(str).tolist()
#     norm_utts = df["normalized_utt"].astype(str).tolist()
#     dists = df["distance_to_centroid"].astype(float).tolist()

#     best_idx = (
#         pd.to_numeric(df["best_idx"], errors="coerce")
#         .astype("Int64")
#         .tolist()
#     )

#     analyzer = MainConceptAnalyzerNormalize(global_cutoff=0.0)

#     # -----------------------
#     # Match logic
#     # -----------------------
#     is_match = [d <= CUTOFF for d in dists]

#     # -----------------------
#     # Matched concepts
#     # -----------------------
#     concepts = analyzer.concepts

#     matched_concepts = []

#     for b, m in zip(best_idx, is_match):

#         if m and b is not None and not pd.isna(b):
#             matched_concepts.append(concepts[int(b)])
#         else:
#             matched_concepts.append(None)

#     # -----------------------
#     # Old-style signals
#     # -----------------------
#     repeated_flags = compute_repeated_flags(
#         analyzer,
#         norm_utts,
#         is_match,
#         SIM_THR
#     )

#     topic_switch_flags = compute_topic_switch_flags(
#         analyzer,
#         utts
#     )

#     # -----------------------
#     # Coherence classification
#     # -----------------------
#     counts, utt_errors = count_errors_fold_one_participant(
#         utterances=utts,
#         is_mainconcept_match=is_match,
#         matched_concepts=matched_concepts,
#         is_repeated_flags=repeated_flags,
#         topic_switch_flags=topic_switch_flags,
#     )

#     total = len(utts)

#     local_pct, global_pct = calculate_coherence_error_percentages(
#         counts,
#         total
#     )

#     # ======================
#     # PRINT SUMMARY
#     # ======================
#     print("\n==============================")
#     print(" COHERENCE ERROR SUMMARY")
#     print("==============================")

#     print("Parquet :", PARQUET_PATH.name)
#     print("Participant:", df["participant"].iloc[0])
#     print("Label:", df["label"].iloc[0])

#     print("Cutoff:", CUTOFF)
#     print("SimThr:", SIM_THR)

#     print("Total utterances:", total)
#     print("MC matches:", sum(is_match))

#     print(f"Local % : {local_pct:.2f}")
#     print(f"Global %: {global_pct:.2f}")

#     print("\n--- RAW COUNTS ---")

#     for et in ERROR_TYPES:
#         print(f"{et:25s} {counts.get(et,0)}")

#     print("\n--- PERCENTAGES ---")

#     for et in ERROR_TYPES:
#         n = counts.get(et, 0)
#         pct = (n / total * 100) if total else 0
#         print(f"{et:25s} {pct:6.2f}%")


# if __name__ == "__main__":
#     main()
