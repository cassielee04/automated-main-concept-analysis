# Automated Main-Concept Analysis

Research code for automatically measuring main-concept production and discourse coherence in spoken Cinderella narratives. The pipeline processes AWS Transcribe JSON files, segments each transcript into utterances, compares those utterances with 34 reference story concepts, and produces participant-level features for control, aphasia, and dementia cohorts.

> [!IMPORTANT]
> This repository contains research software, not a clinical diagnostic tool. Its outputs require validation and expert interpretation before they are used in a clinical or research conclusion.

## What the pipeline measures

For each participant, the primary pipeline calculates:

- number of utterances;
- mean cosine distance from the Cinderella concept centroid;
- total and unique matched main concepts;
- main-concept match ratios;
- story-sequence disorder, measured as the proportion of out-of-order concept pairs;
- local coherence errors;
- global coherence errors; and
- counts and percentages for each discourse classification:
  - topic switching;
  - missing referent;
  - tangential utterance;
  - propositional repetition;
  - filler;
  - conceptual incongruence; and
  - coherent.

Main-concept matching uses `sentence-transformers/all-mpnet-base-v2`. An utterance is treated as a match when its cosine distance from the stored concept centroid is at or below the configured cutoff. The current cohort pipeline uses a cutoff of `0.8292` for all three cohorts.

## Pipeline overview

```text
AWS Transcribe JSON
        |
        v
Transcript extraction and filler cleanup
        |
        v
Stanza-based utterance segmentation
        |
        v
Sentence-transformer embeddings
        |
        +--> main-concept matching and sequence score
        |
        +--> rule-based coherence classification
        |
        v
Participant-level CSV features
        |
        v
Cross-validation and statistical analysis notebooks
```

The repository supports two related workflows:

1. **Direct cohort extraction** with `src/main.py`, which reads transcript JSON files and writes one participant-level CSV per cohort.
2. **Cached, fold-safe extraction** with `src/precompute.py` and `src/split_extract_data.py`, which caches cutoff-independent utterance measurements and fits the main-concept cutoff using only each training fold.

The second workflow is preferable for cross-validation because it avoids estimating the cutoff from held-out participants.

## Repository layout

```text
.
├── config/
│   ├── story_config.yml                     # 34 Cinderella concepts and lexicon
│   └── cinderella_mainconcept_embeddings.pkl # Precomputed concept embeddings
├── data/
│   ├── cache_utts_v2/                       # Per-participant utterance caches
│   └── classification/                      # Extracted features and model results
├── notebooks/                               # Classification and sensitivity analyses
├── src/
│   ├── main.py                              # Direct participant-feature pipeline
│   ├── precompute.py                        # Build cutoff-independent caches
│   ├── split_extract_data.py                # Fold-specific cutoff and features
│   ├── mainconcept_normalize.py             # Semantic concept matcher
│   ├── coherence.py                         # Coherence-error classification
│   ├── segment_utterance.py                 # Utterance segmentation
│   ├── preprocessing.py                     # AWS JSON loading and cleanup
│   ├── normalize_utterances.py              # Filler removal
│   ├── save_cinderella_embeddings.py        # Build/update concept embeddings
│   └── *_submit.sh / run_main.sh            # Sun Grid Engine job scripts
└── test/                                    # Evaluation and data-preparation scripts
```

## Requirements

- Python 3.10 or later
- A CUDA-capable GPU is recommended, but PyTorch can run the embedding model on CPU
- Internet access on the first run to download the Stanza English resources and the Sentence Transformers model
- Enough memory to load Stanza, spaCy, and `all-mpnet-base-v2` together

The repository does not currently include a locked dependency file. A practical environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  numpy pandas pyarrow pyyaml \
  torch sentence-transformers \
  stanza spacy scikit-learn lexicalrichness \
  matplotlib seaborn umap-learn scipy statsmodels \
  jupyter openpyxl
python -m spacy download en_core_web_sm
```

For CUDA installations, install the PyTorch build appropriate for the machine before installing the remaining packages.

## Configuration required before running

Several paths still reflect the original research computing environment. Review the following settings before running the code on another machine:

1. In `src/features.py`, change `STANZA_RESOURCES_DIR` from the hard-coded `/projectnb/...` location to a writable local or shared directory.
2. In `src/main.py` and `src/precompute.py`, update `COHORTS` if the transcript collection is not stored at `../../Data/transcripts/bank/...` relative to `src/`.
3. The SGE scripts in `src/*_submit.sh` contain cluster-specific project names, Conda paths, GPU requests, and absolute repository paths. Update them before submitting jobs.
4. `src/main.py` imports `ciu`, but `src/ciu.py` is not included in this checkout. The import is unused by the current `process_file` implementation; the missing module must be supplied or that import removed before `main.py` can start.

The source currently mixes `src.*` imports with imports local to `src/`, and several configuration paths are relative to the working directory. For the commands below, run from `src/` and expose the repository root through `PYTHONPATH`:

```bash
cd src
export PYTHONPATH=..
```

## Input data

The cohort scripts expect one directory per cohort:

```text
../../Data/transcripts/bank/
├── aphasia/aws/cinderella/
├── controls/aws/cinderella/
└── dementia/aws/cinderella/
```

Files are discovered with the pattern `*_transcribed.json`. Each file must use the AWS Transcribe structure and include at least:

```json
{
  "results": {
    "transcripts": [
      {"transcript": "Full participant transcript..."}
    ]
  }
}
```

The participant identifier is the portion of the filename before the first underscore. For example, `wright03a_cinderella_transcribed.json` is assigned participant ID `wright03a`.

Raw transcripts are intentionally expected outside this repository. Keep protected or identifiable participant data in an access-controlled location and do not commit it.

## Usage

### 1. Direct cohort feature extraction

After completing the configuration steps above:

```bash
cd src
export PYTHONPATH=..
python main.py --cohort controls
python main.py --cohort aphasia
python main.py --cohort dementia
```

The commands write:

```text
data/classification/3_mainconcept_features_ctr_qsub.csv
data/classification/3_mainconcept_features_aph_qsub.csv
data/classification/3_mainconcept_features_dem_qsub.csv
```

To submit the equivalent GPU job on an SGE cluster, first update `src/run_main.sh`, then run, for example:

```bash
qsub run_main.sh controls
```

### 2. Build utterance caches

The cache stores cutoff-independent values—including the nearest concept index and distance to the concept centroid—so different cutoffs can be evaluated without recomputing every embedding.

```bash
cd src
export PYTHONPATH=..
python precompute.py
```

Outputs are written to `data/cache_utts_v2/`:

- one Parquet file per participant; and
- `_index.parquet`, containing the combined utterance cache.

The current checked-in `_index.parquet` does not have a valid Parquet footer in this checkout. Rebuild it with `precompute.py` before running the fold-extraction step if `pandas.read_parquet` reports a “Parquet magic bytes” error.

### 3. Generate fold-specific features

```bash
cd src
export PYTHONPATH=..
python split_extract_data.py
```

This script:

1. creates five participant-grouped, stratified folds;
2. estimates `mean(distance) + 1.0 * standard_deviation(distance)` on each training fold;
3. applies that cutoff to both the training and test participants; and
4. saves participant-level train/test feature tables under `data/classification/fold_features_traincutoff_designB/`.

The fold count, random seed, cutoff multiplier, and repetition threshold are defined near the top of `src/split_extract_data.py`.

### 4. Run analysis notebooks

Start Jupyter from the repository root so notebook-relative paths resolve consistently:

```bash
jupyter lab
```

The main notebooks are:

- `notebooks/aphasia_dementia_controls_ml.ipynb` — multiclass classification;
- `notebooks/aphasia_dementia_controls_ml_ratios.ipynb` — ratio-focused classification;
- `notebooks/mainconcept_cutoff_sensitivity.ipynb` — cutoff sensitivity;
- `notebooks/diagnostic_overlap_sensitivity.ipynb` — sensitivity to diagnostic overlap; and
- `notebooks/aphasia_controls_mainconcept_update.ipynb` — aphasia/control analysis.

## Main output columns

| Column | Meaning |
| --- | --- |
| `participant` | Participant identifier parsed from the filename |
| `total_utterances` | Number of segmented utterances |
| `distance_to_centroid` | Mean utterance distance from the story-concept centroid |
| `num_unique_main_concepts` | Number of distinct matched concepts |
| `num_total_main_concepts` | Number of matches, including repetitions |
| `unique_main_concept_match_ratio` | Unique concepts divided by the 34 reference concepts |
| `total_main_concept_match_ratio` | Total matches divided by total utterances |
| `sequence_score_mainconcept` | Proportion of matched concept pairs occurring out of reference order |
| `local_coherence_mainconcept` | Percentage classified as missing referent or topic switching |
| `global_coherence_mainconcept` | Percentage classified as tangential, repeated, filler, or conceptually incongruent |
| `num_<error_type>` | Count for a discourse classification |
| `ratio_<error_type>_percent` | Percentage for a discourse classification |

Cached utterance files also include `cohort`, `label`, `utt_idx`, `utterance`, `normalized_utt`, `best_idx`, and `error_type`.

## Method notes

- The reference concepts and their order are defined in `config/story_config.yml`.
- Stored reference embeddings are in `config/cinderella_mainconcept_embeddings.pkl`.
- Filler normalization is rule based.
- Utterance segmentation uses Stanza dependency parses to identify main clauses and imperatives.
- Repetition is identified when a matched utterance has cosine similarity greater than `0.8` to a previously stored matched utterance.
- Topic switching uses a similarity threshold of `0.2` between adjacent utterances.
- Missing referents, personal-story tangents, fillers, and conceptual incongruence are classified with rules in `src/coherence.py` and `src/missing_ref_hardcoded.py`.
- Results can change with model, library, preprocessing, path, or threshold changes. Record the environment and configuration when producing reportable results.

## Testing and validation

The `test/` directory currently contains evaluation scripts, not a self-contained automated unit-test suite. A lightweight syntax check is:

```bash
python -m py_compile src/*.py test/*.py
```

Matching performance can be compared with the manually reviewed CSV files under `data/` using the scripts and notebooks in `test/` and `src/notebooks/`. Review all path constants before running those utilities.

## Known limitations

- Dependencies and model versions are not pinned.
- Several scripts depend on the current working directory.
- Resource, transcript, and scheduler paths are tailored to the original cluster environment.
- Some scripts and notebooks are exploratory and contain hard-coded filenames.
- The coherence categories are produced by heuristics and should not be interpreted as ground truth.
- No license is currently included in the repository.

