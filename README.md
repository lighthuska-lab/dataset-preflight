# dataset-preflight

Check local JSONL text datasets before a training or evaluation run. Find malformed
records, duplicates, and exact normalized-text overlap between splits, then save
a file-level SHA-256 manifest with the findings.

Python 3.11 or newer is required. There are no third-party runtime dependencies,
model downloads, cloud services, or API keys. This is an original project by
Light Huska, using public synthetic fixtures rather than employer code or data.

## Try the checked-out source

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
dataset-preflight \
  --train examples/train.jsonl \
  --eval examples/eval.jsonl \
  --id-field id \
  --output results/preflight.json
```

The example files are synthetic and should pass. To check several files in the
same split, repeat `--train` or `--eval`. At least one training file is required.
Use `--text-field content` for a different top-level text key. IDs are checked
only when `--id-field` is supplied.

The CLI prints JSON to stdout and optionally saves the same report atomically.
Exit codes are `0` for no findings, `1` for data findings, `2` for invalid options
or I/O failures, and `130` for interruption. Inputs are never rewritten or
uploaded. Output paths that overlap input paths are rejected before scanning.

## Checks and report interpretation

Each physical line must be a UTF-8 JSON object with a nonempty string in the text
field. Blank lines, BOM-prefixed input, repeated object keys, nonstandard numeric
constants such as NaN, and malformed Unicode text are rejected. A final newline
is optional. CRLF line endings are accepted.

If requested, the ID field must be a nonempty string or integer, excluding
booleans. IDs are compared with their types, so integer `1` and string `"1"` are
different. Only structurally valid records enter duplicate and overlap checks.

Text comparison uses Unicode NFC and collapsed whitespace. It preserves case
and punctuation. SHA-256 fingerprints of that normalized text identify repeated
content within a split and overlap across train/evaluation splits. This is **not
semantic near-duplicate detection** and cannot establish that an evaluation set
is free of every kind of leakage.

The report includes:

- A SHA-256 digest of the raw bytes read from each file, plus its byte/line counts.
- Structurally valid and invalid record counts. Duplicates can still be
  structurally valid, so findings and invalid-record counts are different.
- Per-code finding counts and locations, with the first matching location for
  duplicates or overlap. A record can have more than one finding.
- At most `--max-findings` locations (default 100). Full finding counts remain
  available even when the location list is truncated.

No record text or ID values appear in reports. File paths and line numbers do,
so inspect reports before sharing them. File digests describe bytes actually read,
not an immutable snapshot. Do not edit inputs while validation runs.

Files are read line by line, but the largest line and an index of unique valid
text/IDs must fit memory. This first version targets local text datasets. It has
no general JSON Schema validation, parallel/distributed scan, compressed-file
reader, fuzzy matching, PII detection, quality scoring, or automatic repair.

An empty file is a finding even though it contains zero invalid records. Any
finding makes `ok` false. A failed or interrupted scan does not promise a partial
report. Raw file hashes change with whitespace and line endings even when
normalized text fingerprints match.

## Develop and collaborate

```bash
python -m pip install -e . -r requirements-dev.txt
ruff check .
ruff format --check .
python -m unittest discover -s tests -v
python -m build
```

CI runs offline tests and builds on Python 3.11–3.14. Reviewers should inspect the
normalization contract, cross-split accounting, Unicode handling, and output-path
protection. Use a real issue and focused branch, keep actual contributor identity
on commits, and obtain genuine review before merge. Co-author trailers require
actual shared work and agreement from the contributor.

This project is not published to PyPI. The initial implementation is proposed in
a feature PR; do not treat a green CI result as independent human approval.

## License

[MIT](LICENSE). The fixtures are small synthetic examples. No private datasets
or résumé contact details are included.
