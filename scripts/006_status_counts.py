import pandas as pd

manifest = pd.read_csv(
    "data/catalog/fetch_manifest.csv"
)

print("\nDetected types:")
print(
    manifest["detected_type"]
    .value_counts(dropna=False)
)

print("\nExtraction statuses:")
print(
    manifest["extraction_status"]
    .value_counts(dropna=False)
)

print("\nFetch methods:")
print(
    manifest["fetch_method"]
    .value_counts(dropna=False)
)

print("\nFailed sources:")
failed = manifest[
    manifest["extraction_status"]
    == "fetch_error"
]

if failed.empty:
    print("None")
else:
    print(
        failed[
            [
                "source_id",
                "original_url",
                "error",
            ]
        ].to_string(index=False)
    )