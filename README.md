# Herbarium Label Reader

## Fork attribution and original research

**This is an independently maintained fork of
[Atlas8008/herbarium_label_reader](https://github.com/Atlas8008/herbarium_label_reader),
the original Herbarium Label Reader developed by Matthias Körschens and upstream
contributors.** Credit for the original software, research methodology, and
benchmark belongs to their respective authors and data providers. This fork
adapts that work to Inspect; it is not the authors' original implementation or
an endorsed reproduction of their published results.

Please cite the original paper when building on this work:

> Körschens, Matthias; Bucher, Solveig Franziska; Ritz, Christiane M.; Gebauer,
> Sebastian; Wesenberg, Jens; and Römermann, Christine (2026). **Large language
> vision models for zero-shot handwriting recognition of historical herbarium
> labels.** *Ecological Informatics*, **94**, 103656.
> [doi:10.1016/j.ecoinf.2026.103656](https://doi.org/10.1016/j.ecoinf.2026.103656)
> ([publisher page](https://www.sciencedirect.com/science/article/pii/S1574954126000622)).

The specimen scans and reference labels are provided by **Herbarium
Senckenbergianum Görlitz (GLM), Senckenberg Museum für Naturkunde Görlitz**:
*Herbarium specimens scans (GLM) and associated label data used for zero-shot
handwriting recognition of historical herbarium labels* (2025), version v1,
Zenodo. [doi:10.5281/zenodo.17714208](https://doi.org/10.5281/zenodo.17714208).
Please also cite this dataset when using it.

This fork replaces the original detection and extraction framework with a
whole-image Inspect task and changes the prompt and scoring rules. See
[Migration](#migration) for the differences. For reproducible reports, cite the
paper and dataset and identify this fork's exact Git commit, model, prompt,
image resolution, and scoring policy. [CITATION.cff](CITATION.cff) supplies
machine-readable citations. The original MIT copyright and license notice are
preserved unchanged; the dataset has a separate license described below.

## Inspect implementation

A minimal [Inspect](https://inspect.aisi.org.uk/) evaluation: one whole specimen
image → one model response → deterministic field scores. No detection, cropping,
agents, tools, or model-based grading.

## Setup

Use **Python 3.14.7**, pinned in `.python-version`, and
[uv](https://docs.astral.sh/uv/getting-started/installation/) to install the
locked environment. `pyproject.toml` declares the supported Python 3.14 series
and direct dependencies; `uv.lock` pins their transitive dependencies and hashes.
Run these commands from the repository root:

```bash
uv sync --locked
source .venv/bin/activate
python --version  # Python 3.14.7
```

`uv sync --locked` downloads the pinned Python if needed, creates or updates
`.venv`, installs this project and its Inspect extension, and removes packages
outside the lockfile. This replaces an older Python 3.11 environment and removes
leftover Gradio/Hydra packages. If your shell was already activated, run
`deactivate` before syncing and activate `.venv` again afterward.

Use a current uv release (this setup was resolved with uv 0.12.17). If an older
uv reports that Python 3.14.7 cannot be downloaded, update uv using the same
method you installed it with (`uv self update` for the standalone installer).
Alternatively, use `uv tool run --from uv==0.12.17 uv sync --locked` without
replacing your installed uv.

After pulling repository updates, rerun `uv sync --locked`. You can also run
commands without activation, for example `uv run --locked inspect view`.
`requirements.txt` remains an editable pip-install compatibility entry point,
but it does not enforce `uv.lock`; use uv for reproducible runs.

To intentionally update dependencies, edit the relevant pins in `pyproject.toml`,
then run:

```bash
uv lock
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

Use `uv lock --upgrade` only when you intend to refresh transitive dependencies
as well. Commit `pyproject.toml`, `.python-version`, and `uv.lock` changes together
as applicable. Keep `.venv` untracked.

Run commands from the repository root. Set `OPENROUTER_API_KEY` in your shell or in
`.env` (see `.env.example`). Inspect loads `.env`; exported values take precedence.
Keep real credentials out of
tracked files and command-line model arguments.

## Run

With the local GLM dataset:

```bash
inspect eval herbarium.py \
  -T dataset_path=/Users/liann/Downloads/GLM_scans_mini \
  --model openrouter-cost/openai/gpt-5.6-luna \
  --reasoning-effort medium \
  --limit 1
```

Remove `--limit 1` to evaluate the entire list. This runs paid inference. Inspect
handles concurrency, retries, sample selection, and logs. Use `--max-connections 1`
for sequential model requests, or a larger value for concurrent independent samples.
Use `--sample-id 'GLM-15793_Salix_×_doniana.jpg'` to select a specific specimen.

The default list is `data/handwritten.txt`. Each filename is joined to `Bildname`
in `<dataset_path>/handwritten/label_data.csv`, preserving list order. The image
comes from the same `handwritten` directory. For printed labels, add
`-T image_list=data/printed.txt`; the directory is derived from the list's stem.
Missing images, missing references, duplicate filenames, and malformed reference
rows fail before inference. The image list defines the evaluation population;
extra rows in the reference CSV are ignored.

Whole images are oriented from EXIF, resized in memory to a maximum side of 2048
pixels without upscaling, and sent as JPEG. Change this with `-T max_size=4096`.
Images are prepared when the task is loaded, before Inspect applies `--limit`, so
loading a large list takes time even for a one-sample run. No resized files are
saved. Filenames, EXIF metadata, and ground-truth text are not included in model
messages; sample IDs and reference answers remain in the evaluation logs.

All inference goes through OpenRouter. Use `openrouter-cost/<OpenRouter model ID>`,
for example `openrouter-cost/google/gemini-2.5-pro`. The small adapter in
[openrouter_cost.py](openrouter_cost.py) extends Inspect's OpenRouter provider to
preserve the charge returned by the API. `uv sync --locked` registers
this extension with Inspect (rerun it when updating an existing checkout). It uses `OPENROUTER_API_KEY` and supports
Inspect's OpenRouter options, including `OPENROUTER_BASE_URL`.
Model availability and reasoning settings depend on the selected model.

## Results and scoring

```bash
inspect view
```

Inspect writes `.eval` files to `logs/`, with specimen inputs, model responses,
reference targets, per-field scores, timing, and token usage. There is no custom
CSV output or billing estimator.

### Recording cost

Cost is recorded automatically with the `openrouter-cost/` provider. OpenRouter
returns [`usage.cost`](https://openrouter.ai/docs/cookbook/administration/usage-accounting),
the amount charged to your OpenRouter account, in each response. For regular
OpenRouter calls this becomes `total_cost`. For BYOK calls (`is_byok: true`),
`total_cost` is OpenRouter's charge plus the separately returned
`cost_details.upstream_inference_cost`. The upstream amount is never added for
regular routing, where that would double-count costs. Inspect saves totals in each
sample's `output.usage` and `model_usage`, and the run's `stats.model_usage`.
Both components are preserved in `output.metadata.openrouter_billing`. No pricing file, token-based
calculation, or extra API request is needed. The raw response retains the billing
details, including any separate upstream inference cost.

Use `inspect view` to inspect the logs. Do not pass `--model-cost-config`: Inspect
would overwrite the returned charge with its calculated estimate. The plain
`openrouter/` provider currently drops the returned cost in the pinned Inspect
version; the `openrouter-cost/` prefix enables this adapter.

A returned zero is recorded as zero. Missing or invalid costs remain unknown and
produce a warning; aggregated totals may then be incomplete. These logs account
for returned successful generations, not an account-wide bill or credit-purchase
fees. Missing BYOK billing components remain unknown rather than being treated as zero. Inspect's live cost-limit enforcement in
this version relies on configured prices, so do not rely on `--cost-limit` with
this adapter; set spending limits in OpenRouter instead.

### Grading rules (version 3; unchanged version-2 prompt)

The prompt requests nine fields: `Species name`, `Species author`, `Collection date`,
`Collector's name`, `Country`, `State`, `Location`, `Region`, and `Notes`. All are
strings except `Location` and `Notes`, which are lists with one complete fact per
item. Unknown or absent model answers use `""` or `[]`.

| Field | Full-credit rule |
|---|---|
| Species name | Complete scientific name, including hybrid parents, scored independently of author. Normalize case, whitespace, `×`/`x`, and `ssp.`/`subsp.` or `v.`/`var.` rank abbreviations; no inferred synonyms. |
| Species author | Separate match ignoring abbreviation periods and spacing, or an explicitly approved alternative. Parentheses and author spelling are retained. |
| Collection date | Normalized German/ISO date with exactly the annotated precision. Missing or invented date components fail full credit. |
| Collector's name | The written name or an explicitly approved alternative; no automatic expansion of initials. |
| Country / State | Separate matches for explicitly written information. |
| Location / Notes | Every annotated fact recovered, with no unmatched or repeated items. Fact-list order and ordinary punctuation do not matter. Word order within a fact, negations, numeric signs and decimal values are retained. Other wording requires explicit alternatives. |
| Region | The written region or an explicitly approved alternative; no geographical inference. |

`valid_json` checks the complete schema. Usable fields still receive content credit
when another field is missing, malformed or duplicated, or extra keys are present.
A duplicated field is not graded as a usable answer. Invalid JSON is not repaired.
Missing keys do not count as correct empty answers, even for absent reference fields.

The log includes per-field accuracy, `field_accuracy` (the mean across assessable
fields within each specimen), date year/month/day diagnostics, and Location/Notes
fact precision and recall. Precision is unscored when no facts are predicted;
recall is unscored when no facts are expected. `omitted_facts` and
`unsupported_additions` are counts per specimen (averaged over the run); a wrong
scalar value counts as one omitted expected value and one unsupported prediction.
An extra duplicate fact counts as an unmatched prediction. Error details are saved
in each score's metadata. “Unsupported” means unsupported by the reference, not a
verified hallucination while the catalogue remains provisional.

`specimen_exact` requires all nine fields to have assessable references and all nine
content scores to pass. It is unscored for partially annotated specimens or incomplete fact references. Schema
compliance is separate: a content-perfect answer with an extra key can pass
`specimen_exact` while failing `valid_json`.

### Human-checked answer keys (optional)

Use `-T golden_file=/path/to/goldens.json` to override catalogue fields with reviewed
answers. [goldens.example.json](goldens.example.json) illustrates the format with a
**synthetic example, not a verified transcription**. Replace its filename and
annotations with your own; nothing in this file is loaded by default.

Each annotation has an explicit status:

- `present`: requires a string `value`, or `facts` for Location/Notes. Optional
  `alternatives` list reviewed equivalent strings for that value or individual fact.
- `absent`: a correctly typed empty answer earns credit; nonempty output is an addition.
- `unknown`, `unreadable`, or `uncertain`: excluded from accuracy and addition/omission
  counts. Predictions remain in the log for review.

The JSON has `schema_version: 2` and a `samples` object keyed by image filename.
You may override individual fields; other fields fall back to the current CSV.
Set a field to `unknown` explicitly to exclude a known-bad catalogue answer.
For a nonexhaustive list of known facts, set `complete: false` on Location or Notes.
The grader reports recall but excludes full-field accuracy and precision; unmatched
predictions are recorded as `unverified`, not counted as unsupported additions.
Reviewed fact references default to complete unless explicitly marked otherwise.
Annotations are validated before inference and embedded in the saved targets.

Without overrides, the existing CSV still works as **provisional reference data**.
Blank cells become `unknown`, never `absent`. Country/state split at the first colon;
the species parser separates parenthesized authorities, hybrid formulas, and
infraspecific ranks, including authorities interleaved between ranks. Unrecognized
or ambiguous syntax makes species and author unknown, preserving the raw text for
review, instead of inventing a misleading split. Catalogue Notes are always treated
as incomplete because plant remarks do not exhaust the label text requested by the
prompt. Missing notes therefore do not establish absence.
Location/Notes split at catalogue `/`, `:`, `;`, and `,` delimiters into provisional
facts. These mechanical conversions cannot establish what is written on the label:
curation should correct enriched names, inferred geography, ambiguous authorities,
and fact boundaries. Each field's provenance is recorded in the log.

Grading version 3 leaves the version-2 prompt and output schema unchanged.
Existing logs are untouched. When rescoring saved version-2 targets, the scorer
upgrades catalogue-derived species/author splits and incomplete Notes in memory;
reviewed overrides are preserved. New targets record reference conversion version 3.
Rescore all compared runs with the same grader before comparing their scores.
Version-1 outputs used a different prompt and are not directly comparable.

## Migration

This replaces the previous Hydra runner, Gradio application, custom model
adapters, DINO preprocessing, multi-specimen prompts, CSV parser, and evaluator
with one Inspect task and one scorer. The regression plots, embedding metrics,
Slurm launchers, and heavyweight ML dependencies have been removed. Original
image lists, existing output files, and Git history remain available.

Old commands (`extract_data.py`, `app.py`, and `evaluate.py`) no longer apply.
Old runs are available in `outputs/`; the previous implementation is preserved in
commit `611d435`. New scores are not directly comparable with old scores: the old
evaluator compared only the first two species-name words, date year, and final
collector-name word. The prompt, response format, and scoring policy also changed.

The task and grading code is `herbarium.py`, `scoring.py`, and `prompt.txt`;
`openrouter_cost.py` preserves provider-returned charges. Optional
future analysis or CSV export can use Inspect's [log API](https://inspect.aisi.org.uk/eval-logs.html).

## Offline tests

```bash
uv run --locked python -m unittest discover -s tests -v
```

Tests exercise image preparation, reference joins, strict JSON parsing,
normalization, blank-reference handling, provider credential separation, and a
complete Inspect evaluation with mock responses and saved usage. No paid model
calls are made.

## License

The software remains under the **MIT License**; see [LICENSE](LICENSE). The
upstream notice, **Copyright (c) 2025 Atlas**, and the complete permission and
warranty terms are retained unchanged. Include that notice and license when
redistributing copies or substantial portions of the software. Citation is
requested for scientific credit, not added as a new restriction on the MIT license.

The GLM scans and associated reference data are separately licensed **CC BY-SA
4.0**, as recorded in the [Zenodo metadata](https://zenodo.org/api/records/17714208).
They are not relicensed under MIT. When sharing these materials, retain source
and creator attribution, link the [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/),
and identify changes. Shared adaptations must use CC BY-SA 4.0 or a compatible
license. This includes resized specimen images embedded in shared Inspect logs;
the task applies EXIF orientation, whole-image resizing, and JPEG conversion.
The dataset itself is downloaded separately, and evaluation logs are Git-ignored.

The dataset documentation notes that date and name entries were not corrected
back to literal label text. Account for this when interpreting exact-match
scores or publishing claims about transcription errors.

Inspect and other installed dependencies retain their own licenses. This
repository references them as dependencies rather than vendoring their source;
retain their applicable notices if distributing an environment or application
bundle containing them.
