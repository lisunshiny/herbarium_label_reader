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

Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run commands from the repository root. Set `OPENAI_API_KEY` in your shell or in
`.env` (see `.env.example`). Inspect loads `.env`; exported values take precedence.
OpenRouter uses its separate `OPENROUTER_API_KEY`. Keep real credentials out of
tracked files and command-line model arguments.

## Run

With the local GLM dataset:

```bash
inspect eval herbarium.py \
  -T dataset_path=/Users/liann/Downloads/GLM_scans_mini \
  --model openai/gpt-5.6-luna \
  -M responses_api=true \
  --reasoning-effort medium \
  --model-cost-config pricing.yaml \
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

For OpenRouter, use Inspect's provider syntax (slashes, not the old colon prefix):

```bash
inspect eval herbarium.py \
  -T dataset_path=/Users/liann/Downloads/GLM_scans_mini \
  --model openrouter/google/gemini-2.5-pro \
  --limit 1
```

Model support and account access depend on the provider. Native Inspect provider
settings apply, including `OPENAI_BASE_URL` and `OPENROUTER_BASE_URL` if set.
See [provider configuration](https://inspect.aisi.org.uk/providers.html).

## Results and scoring

```bash
inspect view
```

Inspect writes `.eval` files to `logs/`, with specimen inputs, model responses,
reference targets, per-field scores, timing, and token usage. There is no custom
CSV output or billing estimator.

### Recording cost

Pass `--model-cost-config pricing.yaml` to have Inspect calculate USD cost estimates
and save `total_cost` with sample and run model usage in the `.eval` log. This uses
Inspect's [native cost configuration](https://inspect.aisi.org.uk/setting-limits.html),
not a project-specific calculator. Without configured pricing, cost can be absent
even when token counts are recorded. Existing logs are not retroactively modified.

[pricing.yaml](pricing.yaml) records the source URL, verification date, and rates
for direct OpenAI `gpt-5.6-luna`: $0.20 input, $0.25 cache writes, $0.02 cache reads,
and $1.20 output per million tokens. These are the
[Standard short-context rates](https://developers.openai.com/api/docs/pricing).
Update/add provider-qualified entries for other models or pricing changes;
OpenRouter and other service tiers may have different rates.

Inspect separates regular input (`I`), cache writes (`CW`), and cache reads (`CR`).
Reasoning tokens (`R`) are already included in output (`O`), so they must not be
charged twice. For example, `I=3, CW=3344, CR=0, O=577` estimates **$0.001529**.
These are usage-based estimates, not billing receipts; provider charges remain
authoritative. Keep the pricing-file version with the run's reproducibility record.

The prompt in `prompt.txt` requests exactly seven string-valued JSON fields:

| Output field | Reference column |
|---|---|
| Species name | Spezies Label |
| Collection date | Sammeldatum |
| Collector's name | Sammler |
| Country/State | Geographische_Zuordnung |
| Location | Fundort Label |
| Region | Naturraum |
| Notes | Bemerkung_zur_Pflanze |

`valid_json` measures whether the response is a JSON object with exactly these
keys and string values. Fences, missing/extra keys, duplicate keys, non-string
values, and invalid JSON fail format validation. Invalid responses score zero
on every field that has a nonblank reference.

Each field reports the mean exact-match score across its nonblank references.
Normalization is limited to Unicode NFC, case, whitespace, the hybrid `×`/`x`
symbol, and German month-name dates converted to ISO. Full species names,
collector names, locality details, and dates are compared. Author abbreviations
are retained; synonyms and paraphrases are not equated. Partial dates retain
their precision (`YYYY` or `YYYY-MM`); a full date does not equal a year-only
reference. No translation is requested.

**Blank reference fields are unknown and excluded from that field's accuracy.**
Inspect records their unscored counts. Predictions for those fields stay in logs
for manual review. They are neither credited as correct nor labelled hallucinations.
This strict baseline can penalize legitimate alternate wording, synonyms, and
information absent from the physical label but added to the reference catalogue.

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

The maintained code is `herbarium.py`, `scoring.py`, and `prompt.txt`. Optional
future analysis or CSV export can use Inspect's [log API](https://inspect.aisi.org.uk/eval-logs.html).

## Offline tests

```bash
python -m unittest discover -s tests -v
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
