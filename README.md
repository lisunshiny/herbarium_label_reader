# Herbarium Label Reader

Evaluate vision models on historical herbarium labels using
[Inspect AI](https://inspect.aisi.org.uk/). Each sample sends a whole specimen
image to a model and scores its structured transcription against reference fields.

This project builds on the research and
[original implementation](https://github.com/Atlas8008/herbarium_label_reader)
by Matthias Körschens and collaborators, using specimen scans and catalogue data
from Herbarium Senckenbergianum Görlitz (GLM). This fork replaces the evaluation
pipeline with Inspect AI and introduces a different prompt, revised golden
references, and deterministic field scoring. Its scores are not directly
comparable with the original paper's results.

## Original research and data

Please cite both the paper and dataset when using this work:

- Matthias Körschens, Solveig Franziska Bucher, Christiane M. Ritz, Sebastian
  Gebauer, Jens Wesenberg, and Christine Römermann (2026).
  **Large language vision models for zero-shot handwriting recognition of
  historical herbarium labels.** *Ecological Informatics*, 94, 103656.
  [Paper](https://doi.org/10.1016/j.ecoinf.2026.103656).
- Herbarium Senckenbergianum Görlitz (GLM) (2025).
  **Herbarium specimens scans (GLM) and associated label data used for zero-shot
  handwriting recognition of historical herbarium labels**, v1.
  [Dataset](https://doi.org/10.5281/zenodo.17714208).

The original researchers and data providers supplied the foundation for this
work: the study, specimen dataset, catalogue references, and upstream software.
The Inspect implementation and reference revisions are changes made in this fork.
[CITATION.cff](CITATION.cff) contains the citations. When reporting results from
this fork, also record its Git commit and the exact reference snapshot used.

## Setup

Install [uv](https://docs.astral.sh/uv/) and download and extract
[GLM_scans_mini.zip from Zenodo](https://zenodo.org/records/17714208).
From the repository root:

```bash
uv sync --locked
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env` or your shell. The environment uses Python
3.14.7, pinned in `.python-version`, and dependencies pinned in `uv.lock`.

The extracted dataset directory should contain `handwritten/` and `printed/`,
each with its specimen images and `label_data.csv`. The lists in `data/` select
100 handwritten and 100 printed specimens.

## Run an evaluation

Set `MODEL_ID` to an OpenRouter vision model ID, then run:

```bash
export MODEL_ID='provider/model-name'
uv run --locked inspect eval herbarium.py \
  -T dataset_path=/path/to/GLM_scans_mini \
  -T golden_file=/path/to/handwritten-goldens.json \
  --model "openrouter-cost/$MODEL_ID" \
  --limit 1
```

Replace the placeholder paths and model ID. Remove `--limit 1` for the full split.
For printed labels, add `-T image_list=data/printed.txt` and supply the printed
reference file. Omitting `golden_file` uses the original catalogue through a
provisional conversion; it does **not** reproduce runs using the revised goldens.

Images are EXIF-oriented and resized in memory to a maximum side of 2048 pixels,
without upscaling. Set `-T max_size=4096` to change that limit. The model receives
the image and [prompt](prompt.txt); filenames and reference answers are not sent
as text. Image preparation covers the selected list before Inspect applies
`--limit`.

Inspect handles inference, concurrency, retries, and logs. To view results:

```bash
uv run --locked inspect view
```

The `openrouter-cost/` provider records API-returned charges in Inspect logs,
including separately reported upstream costs for BYOK requests. Missing billing
information remains unknown. Do not use `--model-cost-config`, which replaces
returned charges with estimates, or rely on `--cost-limit` with this adapter.

## Revised golden references

The original catalogue is the starting point for our reference revisions. As the
[dataset documentation](https://zenodo.org/records/17714208) explains, catalogue
entries can include information beyond the literal label, and names and dates
were not fully reverted to the written text. Our prompt asks for information
supported by the label, so we revised reference fields to match that task.

The current working goldens combine human-approved corrections and AI-reviewed
annotations, with image and OCR evidence used during review. Model agreement
was used in preparing some annotations. These references should not be described
as an independently human-transcribed gold standard. Notes are marked unknown
in the current run snapshots and excluded from content scoring, although the
prompt still requests them.

The working goldens and review tooling are local and are not distributed in this
repository. [goldens.example.json](goldens.example.json) documents the format
with a synthetic example. To reproduce a run, retain its exact golden file,
review provenance, and file hash alongside the code revision and model settings.

Golden files use `schema_version: 2`, with samples keyed by image filename.
Each field has one of these statuses:

- `present`: an expected value or list of facts, optionally with accepted alternatives.
- `absent`: the answer must be empty.
- `unknown`, `unreadable`, or `uncertain`: excluded from content scoring.

Overrides apply field by field; unspecified fields fall back to the catalogue.
Blank catalogue values become unknown, not confirmed absent. Location and Notes
can use `complete: false` for partial references: recall is scored, but full-field
accuracy and precision are excluded. References are validated before inference
and embedded in the evaluation log.

## Scoring

[scoring.py](scoring.py) implements grader **version 5**. The output schema has
nine fields: species name, species author, collection date, collector, country,
state, location, region, and notes. Location and Notes are lists; the other fields
are strings.

The grader uses field-specific normalization and explicit accepted alternatives.
It checks the full species name separately from its author and preserves the
precision of collection dates. Location matching tolerates split or combined
facts, recognized MTB grid and elevation notation, and a fixed list of habitat
and soil terms. Required locality facts and unmatched additions still affect the
score. The exact normalization and context allowlist are defined in the scorer.

The main metric, `field_accuracy`, averages assessable field scores within each
specimen; Inspect then reports the mean across specimens. Logs also include
per-field accuracy, JSON validity, date components, fact precision and recall,
and counts of omissions and unsupported additions. “Unsupported” means absent
from the reference, which may itself be incomplete or incorrect.

`specimen_exact` requires complete, assessable references for all nine fields.
It is unscored for the current goldens because Notes are excluded. JSON validity
is scored separately from content accuracy. Compare runs using the same prompt,
reference snapshot, image settings, and grader version.

## Development and results

The evaluation lives in [herbarium.py](herbarium.py), the extraction instructions
in [prompt.txt](prompt.txt), and the billing adapter in
[openrouter_cost.py](openrouter_cost.py). Run the offline tests with:

```bash
uv run --locked python -m unittest discover -s tests -v
```

Raw evaluation logs, local run artifacts, and temporary review tools are ignored
by Git. Results added to the repository should be curated summaries identifying
the models, sample counts, settings, code revision, grader version, and reference
snapshot, with a link to archived logs when available.

## License

Code is licensed under [MIT](LICENSE), retaining the original
Copyright (c) 2025 Atlas notice. GLM scans and associated label data have a separate
[CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).
Derived reference annotations retain the dataset attribution and license.
