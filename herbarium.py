"""One specimen image per Inspect sample; no tools, agents, or label detection."""
import base64
import csv
import io
import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from inspect_ai.solver import generate
from PIL import Image, ImageOps

from scoring import FIELDS, OUTPUT_FIELDS, label_fields, legacy_reference, validate_reference

ROOT = Path(__file__).resolve().parent
PROMPT = (ROOT / "prompt.txt").read_text(encoding="utf-8")


def image_input(path: Path, max_size: int) -> str:
    """Resize the whole image in memory and omit filenames and EXIF from model input."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        image.thumbnail((max_size, max_size))
        image = image.convert("RGB")
        with io.BytesIO() as buffer:
            image.save(buffer, format="JPEG")
            return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def load_dataset(dataset_path: str, image_list: str, max_size: int, golden_file: str | None = None) -> MemoryDataset:
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    listing = Path(image_list).expanduser().resolve()
    directory = Path(dataset_path).expanduser().resolve() / listing.stem
    truth_path = directory / "label_data.csv"
    with truth_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Bildname", *FIELDS.values()}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Ground truth is missing columns: {', '.join(sorted(missing))}")
        rows = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed ground-truth row at line {reader.line_num}")
            name = row["Bildname"]
            if name in rows:
                raise ValueError(f"Duplicate ground-truth image: {name}")
            rows[name] = row
    names = [name.strip() for name in listing.read_text(encoding="utf-8-sig").splitlines() if name.strip()]
    if not names:
        raise ValueError("Image list is empty")
    if len(names) != len(set(names)):
        raise ValueError("Image list contains duplicate filenames")
    annotations = {}
    if golden_file:
        document = json.loads(Path(golden_file).expanduser().read_text(encoding="utf-8"))
        if not isinstance(document, dict) or set(document) != {"schema_version", "samples"} or document["schema_version"] != 2:
            raise ValueError("Golden file requires schema_version: 2 and samples")
        annotations = document["samples"]
        if not isinstance(annotations, dict):
            raise ValueError("Golden samples must be keyed by image filename")
        for name, fields in annotations.items():
            if name not in rows:
                raise ValueError(f"Golden image has no catalogue row: {name}")
            if not isinstance(fields, dict) or set(fields) - set(OUTPUT_FIELDS):
                raise ValueError(f"Unknown golden fields for {name}")
            # Validate partial overrides together with provisional catalogue values.
            validate_reference({**legacy_reference(rows[name]), **fields})
    # Validate the join and image paths before doing any image processing or inference.
    for name in names:
        if name not in rows:
            raise ValueError(f"No ground-truth row for {name}")
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f"Missing image or image outside dataset directory: {name}")
    samples = [
        Sample(
            id=name,
            input=[ChatMessageUser(content=[
                ContentText(text=PROMPT),
                ContentImage(image=image_input(directory / name, max_size)),
            ])],
            target=json.dumps({"schema_version": 2, "fields": {
                **legacy_reference(rows[name]), **annotations.get(name, {})}}, ensure_ascii=False),
            metadata={"source_image": name, "max_size": max_size,
                      "golden_fields": list(annotations.get(name, {})),
                      "reference_policy": "golden overrides with provisional catalogue fallback"},
        )
        for name in names
    ]
    return MemoryDataset(samples, name=listing.stem, location=str(truth_path))


@task
def herbarium(
    dataset_path: str,
    image_list: str = str(ROOT / "data" / "handwritten.txt"),
    max_size: int = 2048,
    golden_file: str | None = None,
) -> Task:
    return Task(
        dataset=load_dataset(dataset_path, image_list, max_size, golden_file),
        solver=generate(),
        scorer=label_fields(),
        version=2,
    )
