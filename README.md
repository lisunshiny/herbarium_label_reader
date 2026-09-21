# Herbarium Label Reader

This repo runs the tests behind the
[Herbarium AI Leaderboard](https://huggingface.co/spaces/lisunshiny/herbarium-ai-leaderboard).
Each model gets a specimen image and returns label details as JSON. We compare
those details with reference answers using [Inspect AI](https://inspect.aisi.org.uk/).

## Research and data

This fork builds on research by
[Körschens and colleagues (2026)](https://doi.org/10.1016/j.ecoinf.2026.103656),
published in *Ecological Informatics*, and their
[original code](https://github.com/Atlas8008/herbarium_label_reader).
It uses [specimen scans and label data](https://doi.org/10.5281/zenodo.17714208)
from Herbarium Senckenbergianum Görlitz (GLM).

It adds an Inspect AI pipeline, a different prompt, revised reference answers,
and new scoring rules. These scores are not directly comparable with the paper.
[CITATION.cff](CITATION.cff) contains the citations.

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
  -T golden_file=references/v1/handwritten.json \
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

## Run all 200 specimens

The [v1 reference answers](references/v1/README.md) are included in this repo,
with review records and file hashes. Run both splits with:

```bash
export DATASET_PATH='/path/to/GLM_scans_mini'
export GOLDEN_DIR='references/v1'
export MODEL_ID='provider/model-name'
export RUN_NAME='my-model-run'

for split in handwritten printed; do
  uv run --locked inspect eval herbarium.py \
    -T dataset_path="$DATASET_PATH" \
    -T image_list="data/$split.txt" \
    -T golden_file="$GOLDEN_DIR/$split.json" \
    -T max_size=2048 \
    --model "openrouter-cost/$MODEL_ID" \
    --log-dir "logs/$RUN_NAME/$split" || break
done
```

Change `MODEL_ID` and `RUN_NAME` to test another model. Start with `--limit 1`
on each split to check that it accepts images and returns the expected JSON.
These commands make paid API requests and save local logs. They do not upload
results to the leaderboard.

To repeat a published run, match its code, prompt, reference files, image size,
provider routing, and model settings. These commands use model defaults; a
published run may use specific `--reasoning-effort`, `--temperature`,
`--max-tokens`, or `--max-connections` values. The three models marked with an
asterisk also need the separate code-fence regrading described above. New runs
can give different answers even with the same settings.

When sharing results, include both split scores, completed and failed sample
counts, JSON validity, model settings, code revision, and reference hashes.
With 100 scored specimens per split, the combined score is the mean of the two
split scores. Calculate cost per 100 from the samples with recorded charges,
and state how many charges are missing. Calculate median time across individual
samples from both splits.

## Reference answers

The original catalogue sometimes includes information beyond what is written on
the label. Our prompt asks for information supported by the label, so we revised
the answers used for scoring. The code calls these files “goldens.”

The frozen [v1 references](references/v1/README.md) cover all 200 specimens:

| Labels | Human-approved | AI-reviewed only |
|---|---:|---:|
| Handwritten | 10 | 90 |
| Printed | 70 | 30 |

Review used specimen images, OCR, and model agreement. Some of the models being
tested helped prepare the answers, including answers later approved by a human.
These are not independent human transcriptions. The review groups were not
chosen at random, and most human-approved specimens are printed.

Notes are marked unknown and left out of content scoring, though the prompt
still asks for them. The reference files include review records and hashes;
[goldens.example.json](goldens.example.json) shows the format with a made-up
example. Keep the exact reference files used for each run.

Golden files use `schema_version: 2`, with samples keyed by image filename.
Each field has a status:

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

The grader accounts for formatting differences and listed alternative answers.
It checks the species name separately from its author and keeps dates at the
level of detail written on the label. Location matching tolerates split or combined
facts, recognized MTB grid and elevation notation, and a fixed list of habitat
and soil terms. Missing location details and extra claims still affect the score.
The full matching rules are in the scorer.

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
