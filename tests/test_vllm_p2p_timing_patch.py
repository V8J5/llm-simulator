import ast
import tempfile
import unittest
from pathlib import Path

from scripts.patch_vllm019_p2p_timing import (
    COMPLETE_MARKER,
    START_MARKER,
    instrument_text,
    state,
)


class VLLMP2PTimingPatchTests(unittest.TestCase):
    SOURCE = '''import regex as re

def start_load(self, metadata, forward_context):
        for request in metadata.requests:
            request_id = request.request_id
            for layer_name in forward_context.no_compile_layers:
                kv_cache = recv_tensor(request.request_id + "#" + layer_name)
                inject_kv_into_layer(
                    layer, kv_cache, request.block_ids, request.request_id
                )
'''

    def test_instrumentation_is_complete_idempotent_and_valid(self):
        for source in (self.SOURCE, self.SOURCE.replace(
                "import regex as re", "import re")):
            with self.subTest(import_style=source.splitlines()[0]):
                patched = instrument_text(source)
                self.assertIn(START_MARKER, patched)
                self.assertIn(COMPLETE_MARKER, patched)
                self.assertIn("import time", patched)
                self.assertNotIn("timegex", patched)
                self.assertEqual(instrument_text(patched), patched)
                ast.parse(patched)

    def test_state_rejects_partial_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p2p_nccl_connector.py"
            path.write_text(self.SOURCE + START_MARKER, encoding="utf-8")
            self.assertEqual(state(path), "partial")

    def test_unknown_source_is_not_modified(self):
        with self.assertRaisesRegex(RuntimeError, "unrecognized"):
            instrument_text("different source")


if __name__ == "__main__":
    unittest.main()
