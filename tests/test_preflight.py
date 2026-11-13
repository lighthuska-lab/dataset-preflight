from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset_preflight.checks import InputFile, check_dataset, normalized_text
from dataset_preflight.cli import main


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def data(self, name, records, split="train"):
        path = self.root / name
        path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
        return InputFile(path, split)

    def codes(self, report):
        return report["summary"]["counts"]

    def test_clean_splits_and_byte_fingerprints(self):
        train = self.data("train", [{"text": "Training"}])
        evaluate = self.data("eval", [{"text": "Evaluation"}], "eval")
        result = check_dataset([train, evaluate])
        self.assertTrue(result["ok"])
        self.assertEqual(result["summary"]["valid_records"], 2)
        self.assertEqual(
            result["files"][0]["sha256"],
            hashlib.sha256(train.path.read_bytes()).hexdigest(),
        )
        self.assertEqual(result["files"][0]["bytes"], train.path.stat().st_size)

    def test_whitespace_and_nfc_normalization_preserve_case(self):
        self.assertEqual(normalized_text(" e\u0301\n  word\t"), "é word")
        f = self.data(
            "train",
            [
                {"text": " Hello  world "},
                {"text": "Hello\nworld"},
                {"text": "hello world"},
            ],
        )
        result = check_dataset([f])
        self.assertEqual(self.codes(result), {"duplicate_text": 1})
        self.assertEqual(result["findings"][0]["line"], 2)
        self.assertEqual(result["findings"][0]["first_seen"]["line"], 1)

    def test_duplicates_across_files_in_one_split(self):
        first = self.data("a", [{"text": "same"}])
        second = self.data("b", [{"text": "same"}])
        result = check_dataset([first, second])
        self.assertEqual(self.codes(result), {"duplicate_text": 1})

    def test_cross_split_overlap_in_either_order(self):
        train = self.data("train", [{"text": "same"}])
        evaluate = self.data("eval", [{"text": " same "}], "eval")
        for inputs in ([train, evaluate], [evaluate, train]):
            with self.subTest(inputs=inputs):
                self.assertEqual(
                    self.codes(check_dataset(inputs)), {"cross_split_text": 1}
                )

    def test_within_and_cross_split_findings_are_both_reported(self):
        train = self.data("train", [{"text": "same"}])
        evaluate = self.data("eval", [{"text": "same"}, {"text": "same"}], "eval")
        self.assertEqual(
            self.codes(check_dataset([train, evaluate])),
            {"cross_split_text": 2, "duplicate_text": 1},
        )

    def test_ids_are_optional_and_type_sensitive(self):
        f = self.data(
            "train",
            [{"id": 1, "text": "a"}, {"id": "1", "text": "b"}, {"id": 1, "text": "c"}],
        )
        self.assertTrue(check_dataset([f])["ok"])
        self.assertEqual(
            self.codes(check_dataset([f], id_field="id")), {"duplicate_id": 1}
        )

    def test_cross_split_id_detects_overlap_with_different_text(self):
        a = self.data("train", [{"id": "x", "text": "a"}])
        b = self.data("eval", [{"id": "x", "text": "b"}], "eval")
        self.assertEqual(
            self.codes(check_dataset([a, b], id_field="id")), {"cross_split_id": 1}
        )

    def test_invalid_ids(self):
        f = self.data(
            "train",
            [
                {"id": value, "text": f"text-{index}"}
                for index, value in enumerate([None, True, 1.5, [], {}, " "])
            ],
        )
        self.assertEqual(
            self.codes(check_dataset([f], id_field="id")), {"invalid_id": 6}
        )

    def test_missing_id_when_requested(self):
        f = self.data("train", [{"text": "okay"}])
        self.assertEqual(
            self.codes(check_dataset([f], id_field="id")), {"invalid_id": 1}
        )

    def test_record_shapes_and_text_errors(self):
        f = self.data("train", [[], 1, None, {}, {"text": 4}, {"text": "\t"}])
        result = check_dataset([f])
        self.assertEqual(
            self.codes(result),
            {"not_object": 3, "missing_text": 1, "text_not_string": 1, "empty_text": 1},
        )
        self.assertEqual(result["summary"]["invalid_records"], 6)

    def test_invalid_json_utf8_blank_and_nonstandard_json(self):
        path = self.root / "bad"
        path.write_bytes(
            b'not json\n\xff\n\n{"text":"x","n":NaN}\n{"text":"a","text":"b"}\n'
        )
        result = check_dataset([InputFile(path, "train")])
        self.assertEqual(
            self.codes(result), {"invalid_json": 3, "invalid_utf8": 1, "blank_line": 1}
        )

    def test_surrogate_escapes_are_data_findings_not_crashes(self):
        f = self.data("train", [{"text": "\ud800"}, {"text": "valid", "id": "\ud800"}])
        self.assertEqual(
            self.codes(check_dataset([f], id_field="id")),
            {"invalid_text_unicode": 1, "invalid_id": 1},
        )

    def test_empty_file(self):
        path = self.root / "empty"
        path.write_bytes(b"")
        result = check_dataset([InputFile(path, "train")])
        self.assertEqual(self.codes(result), {"empty_file": 1})
        self.assertEqual(result["files"][0]["lines"], 0)

    def test_final_line_without_newline_and_crlf(self):
        path = self.root / "input"
        path.write_bytes(b'{"text":"a"}\r\n{"text":"b"}')
        result = check_dataset([InputFile(path, "train")])
        self.assertTrue(result["ok"])
        self.assertEqual(result["files"][0]["lines"], 2)

    def test_findings_cap_keeps_actual_counts(self):
        f = self.data("train", [{"text": "same"}] * 5)
        r = check_dataset([f], max_findings=1)
        self.assertEqual(len(r["findings"]), 1)
        self.assertEqual(r["summary"]["finding_count"], 4)
        self.assertTrue(r["summary"]["findings_truncated"])

    def test_no_text_or_identifiers_in_report(self):
        f = self.data(
            "train", [{"id": "private-identifier", "text": "private-text"}] * 2
        )
        encoded = json.dumps(check_dataset([f], id_field="id"))
        self.assertNotIn("private-identifier", encoded)
        self.assertNotIn("private-text", encoded)

    def test_custom_text_field(self):
        f = self.data("train", [{"content": "works"}])
        self.assertTrue(check_dataset([f], text_field="content")["ok"])

    def test_invalid_configuration(self):
        f = self.data("train", [{"text": "works"}])
        cases = [
            {"max_findings": 0},
            {"max_findings": True},
            {"text_field": ""},
            {"id_field": ""},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                check_dataset([f], **kwargs)
        for inputs in (
            [],
            [f, f],
            [InputFile(f.path, "test")],
            [InputFile(self.root / "missing", "train")],
        ):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                check_dataset(inputs)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "input.jsonl"
        self.source.write_text('{"text":"a"}\n', encoding="utf-8")

    def invoke(self, *extra):
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = main(["--train", str(self.source), *map(str, extra)])
        return code, output.getvalue()

    def test_stdout_and_atomic_saved_report(self):
        path = self.root / "reports" / "run.json"
        code, output = self.invoke("--output", path)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output), json.loads(path.read_text()))
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_data_findings_return_one_with_report(self):
        self.source.write_text("invalid\n", encoding="utf-8")
        code, output = self.invoke()
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output)["summary"]["finding_count"], 1)

    def test_source_cannot_be_overwritten(self):
        before = self.source.read_bytes()
        with patch("dataset_preflight.cli.check_dataset") as check:
            code, _ = self.invoke("--output", self.source)
            check.assert_not_called()
        self.assertEqual(code, 2)
        self.assertEqual(self.source.read_bytes(), before)

    def test_nested_and_directory_outputs_fail_before_scan(self):
        for path in (self.source / "report", self.root):
            with (
                self.subTest(path=path),
                patch("dataset_preflight.cli.check_dataset") as check,
            ):
                code, _ = self.invoke("--output", path)
                self.assertEqual(code, 2)
                check.assert_not_called()

    def test_missing_input_preserves_old_report(self):
        path = self.root / "report"
        path.write_text("old")
        self.source.unlink()
        code, _ = self.invoke("--output", path)
        self.assertEqual(code, 2)
        self.assertEqual(path.read_text(), "old")

    def test_module_entrypoint(self):
        r = subprocess.run(
            [sys.executable, "-m", "dataset_preflight", "--train", str(self.source)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(json.loads(r.stdout)["ok"])
