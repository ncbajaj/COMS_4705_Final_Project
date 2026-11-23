import argparse
from pathlib import Path
import unittest

from rarecrash.compare import fmt, parse_run, table


class CompareTests(unittest.TestCase):
    def test_parse_run(self):
        self.assertEqual(parse_run("uniform=runs/a"), ("uniform", Path("runs/a")))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_run("runs/a")

    def test_fmt_single_and_multiple(self):
        self.assertEqual(fmt([0.5]), "0.500")
        self.assertEqual(fmt([0.4, 0.6]), "0.500 ± 0.141")
        self.assertEqual(fmt([None]), "n/a")

    def test_table_has_one_row_per_label(self):
        result = {"all": {"all_fields_adherence": 1.0}}
        lines = table({"retrieval": [result], "model": [result, result]}, "all").splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("| model | 2 | 1.000 ± 0.000 |", lines[3])


if __name__ == "__main__":
    unittest.main()
