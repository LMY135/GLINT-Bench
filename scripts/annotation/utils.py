import json
from pathlib import Path


def load_jsonl(path, key=None):
    """
    Load a JSONL file.

    If key is provided, return a dictionary
    indexed by that field.

    Example:
        load_jsonl(path, key="question_id")
    """

    path = Path(path)

    if not path.exists():
        return {} if key else []

    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {path}, "
                    f"line {line_number}: {e}"
                )

            records.append(record)

    if key is None:
        return records

    indexed = {}

    for record in records:

        if key not in record:
            raise ValueError(
                f"Missing '{key}' in {path}."
            )

        value = record[key]

        if value in indexed:
            raise ValueError(
                f"Duplicate {key}: {value}"
            )

        indexed[value] = record

    return indexed


def save_jsonl(path, records):
    """
    Write records to a JSONL file.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )