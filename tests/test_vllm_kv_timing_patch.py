import tempfile
import unittest
from pathlib import Path

from scripts.patch_vllm019_kv_timing import (
    PROMOTE_MARKER,
    RECV_MARKER,
    WAIT_MARKER,
    instrument_text,
    state,
)


class VLLMKVTimingPatchTests(unittest.TestCase):
    SOURCE = """                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                self._update_waiting_for_remote_kv(request)
                logger.debug("Finished recving KV transfer for request %s", req_id)
"""

    def test_instrumentation_is_complete_and_idempotent(self):
        patched = instrument_text(self.SOURCE)
        self.assertIn(WAIT_MARKER, patched)
        self.assertIn(RECV_MARKER, patched)
        self.assertIn(PROMOTE_MARKER, patched)
        self.assertEqual(instrument_text(patched), patched)

    def test_state_rejects_partial_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scheduler.py"
            path.write_text(self.SOURCE + WAIT_MARKER, encoding="utf-8")
            self.assertEqual(state(path), "partial")

    def test_unknown_source_is_not_modified(self):
        with self.assertRaisesRegex(RuntimeError, "unrecognized"):
            instrument_text("different source")


if __name__ == "__main__":
    unittest.main()
