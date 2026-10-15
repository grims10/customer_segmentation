import unittest

import pandas as pd

from src.data_loader import REQUIRED_COLUMNS, validate_data


class ValidateDataTests(unittest.TestCase):
    def test_accepts_required_schema(self):
        frame = pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
        frame.loc[0] = [None] * len(frame.columns)
        self.assertIs(validate_data(frame), frame)

    def test_reports_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "CustomerID"):
            validate_data(pd.DataFrame({"InvoiceNo": [1]}))

    def test_rejects_empty_data(self):
        with self.assertRaisesRegex(ValueError, "no rows"):
            validate_data(pd.DataFrame(columns=sorted(REQUIRED_COLUMNS)))


if __name__ == "__main__":
    unittest.main()
