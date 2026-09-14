import json
import tempfile
import unittest
from pathlib import Path

from scripts.calibration_roles import (
    FIXED_BATCH_DECODE,
    ONLINE_COLOCATED,
    require_calibration_role,
)


class CalibrationRoleTests(unittest.TestCase):
    def write_config(self, metadata):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8")
        with handle:
            json.dump({"metadata": metadata}, handle)
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return Path(handle.name)

    def test_matching_role_is_accepted(self):
        path = self.write_config({"calibration_role": ONLINE_COLOCATED})
        self.assertEqual(
            require_calibration_role(path, ONLINE_COLOCATED),
            ONLINE_COLOCATED,
        )

    def test_incompatible_role_is_rejected(self):
        path = self.write_config({"calibration_role": FIXED_BATCH_DECODE})
        with self.assertRaisesRegex(ValueError, "role mismatch"):
            require_calibration_role(path, ONLINE_COLOCATED)

    def test_legacy_file_remains_compatible(self):
        path = self.write_config({"method": "legacy"})
        self.assertEqual(
            require_calibration_role(path, ONLINE_COLOCATED),
            "legacy_unspecified",
        )


if __name__ == "__main__":
    unittest.main()
