import pandas as pd

validation = pd.read_csv(
    "data/review/collection_validation.csv"
)

actionable = validation[
    validation["severity"].isin(
        ["warning", "fatal"]
    )
]

if actionable.empty:
    print(
        "No sources require action."
    )
else:
    columns = [
        "source_id",
        "title",
        "detected_type",
        "extraction_status",
        "severity",
        "message",
    ]

    print(
        actionable[columns]
        .to_string(index=False)
    )