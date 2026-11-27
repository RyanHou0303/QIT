import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from src.features.diagnose_features import diagnose_features, load_feature_panel


class FeatureDiagnosticsTests(unittest.TestCase):
    def test_saved_feature_list_excludes_labels_and_intermediate_columns(self):
        panel = pl.DataFrame({
            "raw_mom_4h": [0.01],
            "res_ret_1h": [0.002],
            "fwd_res_ret_z_24h": [1.5],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.parquet"
            panel.write_parquet(
                path, metadata={"feature_cols": json.dumps(["raw_mom_4h"])}
            )
            loaded, feature_cols = load_feature_panel(path)

        self.assertTrue(loaded.equals(panel))
        self.assertEqual(feature_cols, ["raw_mom_4h"])

    def test_old_artifact_requests_rebuild_instead_of_guessing_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.parquet"
            pl.DataFrame({"x": [1.0]}).write_parquet(path)
            with self.assertRaisesRegex(ValueError, "Run src/features/alphas.py"):
                load_feature_panel(path)

    def test_profile_separates_null_nan_and_infinity(self):
        panel = pl.DataFrame({
            "timestamp": [datetime(2023, 1, 1) + timedelta(hours=h) for h in range(5)],
            "x": [None, float("nan"), float("inf"), 1.0, 3.0],
        })
        profile, correlated = diagnose_features(panel, ["x"])
        row = profile.row(0, named=True)
        self.assertEqual(row["valid_count"], 2)
        self.assertAlmostEqual(row["null_ratio"], 0.2)
        self.assertEqual(row["nan_count"], 1)
        self.assertEqual(row["inf_count"], 1)
        self.assertEqual(row["mean"], 2.0)
        self.assertEqual(row["first_valid"], datetime(2023, 1, 1, 3))
        self.assertTrue(correlated.is_empty())


if __name__ == "__main__":
    unittest.main()
