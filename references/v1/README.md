# Leaderboard references v1

Frozen reference snapshot for the Herbarium AI Leaderboard, prepared on
20 September 2026. These files are byte-for-byte copies of the golden files used
by the expanded evaluation runs. Use them with grader v5 and the corresponding
prompt; the evaluation code is recorded at commit `077d392`.

| Split | Specimens | Human-approved | AI-reviewed only |
|---|---:|---:|---:|
| Handwritten | 100 | 10 | 90 |
| Printed | 100 | 70 | 30 |
| Total | 200 | 80 | 120 |

Human approval includes decisions assisted by agreement among Astra, Fable, and
Gemini. The other 120 specimens were reviewed with AI assistance and were not
human-approved. Review used specimen images and, for difficult cases, enlarged
views and Google Vision OCR. These are model-assisted references, not independent
human transcriptions. Some evaluated models contributed to their construction,
so agreement with these references is not an independent assessment of those
models. The two review groups were not randomly selected.

## Contents and use

- `handwritten.json` and `printed.json`: complete field annotations for each split.
- `provenance.json`: source attribution, review categories per specimen, code and
  prompt hashes, and verification against eight regraded evaluation logs.
- `SHA256SUMS`: SHA-256 hashes of both golden files and the provenance file.

From the repository root, pass
`-T golden_file=references/v1/handwritten.json` for handwritten specimens, or
`-T golden_file=references/v1/printed.json` with
`-T image_list=data/printed.txt` for printed specimens. The original dataset is
still required for specimen images and the loader's catalogue join.

All nine fields are explicit, so no field annotations fall back to the catalogue.
Notes are `unknown` throughout and excluded from content scoring. Other uncertain
or unassessed fields also remain excluded. A `human_review` source on an unknown
Notes field is an exclusion marker, not evidence that the specimen was
human-approved. Use the per-specimen categories in `provenance.json` for subsets.
Some reason strings describe proposals as pending approval; these historical
strings are retained verbatim, while the exported source and provenance record
the review category.

The scoring annotations match all 200 targets in each of four earlier model
evaluations (eight split logs). Only explanatory `reason` strings differ between
those regraded targets and this snapshot. Raw logs and working review state are
not included. Exact inference settings, costs, and timing require the individual
run records in addition to these references.

Keep v1 unchanged when revising answers. Publish corrections in a new version
and identify the reference version used for each leaderboard snapshot.

## Attribution and license

These annotations adapt the catalogue and label data in **Herbarium specimens
scans (GLM) and associated label data used for zero-shot handwriting recognition
of historical herbarium labels**, v1 (2025), provided by **Herbarium
Senckenbergianum Görlitz (GLM), Senckenberg Museum für Naturkunde Görlitz**.
[Source dataset](https://doi.org/10.5281/zenodo.17714208).

The underlying research is by Matthias Körschens, Solveig Franziska Bucher,
Christiane M. Ritz, Sebastian Gebauer, Jens Wesenberg, and Christine Römermann:
[Large language vision models for zero-shot handwriting recognition of historical
herbarium labels](https://doi.org/10.1016/j.ecoinf.2026.103656) (2026).

Changes made by Liann Sun with AI-assisted review in this fork include revised
transcriptions, separate species-author and country/state fields, explicit
annotation statuses, accepted alternatives, and exclusion of Notes. These changes
are not attributed to or endorsed by the original researchers.

The reference annotations and accompanying provenance are licensed under
**[Creative Commons Attribution-ShareAlike 4.0 International
(CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)**, separately from
the repository's MIT-licensed code. Retain attribution, identify changes, and
share adaptations under the same license.
