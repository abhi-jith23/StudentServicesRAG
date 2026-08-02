import json
from pathlib import Path

limits = {
    "compact": 220,
    "broad": 350,
}

for name, limit in limits.items():
    path = Path(
        f"data/processed/chunks_{name}.jsonl"
    )

    rows = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    counts = [
        int(row["token_count"])
        for row in rows
    ]

    oversized = [
        row
        for row in rows
        if int(row["token_count"]) > limit
    ]

    print(f"\n{name.upper()}")
    print("Chunks:", len(rows))
    print("Minimum:", min(counts))
    print("Maximum:", max(counts))
    print("Oversized:", len(oversized))