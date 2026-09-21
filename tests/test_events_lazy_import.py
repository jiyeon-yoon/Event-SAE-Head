import subprocess
import sys


def test_metadata_and_score_import_do_not_require_optional_vlm():
    result = subprocess.run([sys.executable, "-c", """
import sys
class RejectOptional:
    def find_spec(self, fullname, *args):
        if fullname.startswith(('google.genai', 'tensorflow', 'transformers')):
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, RejectOptional())
from event_sae.events import load_jsonl, PHASE_LABELS
from event_sae.scoring.score_matrix import score_cluster_features
assert callable(load_jsonl) and callable(score_cluster_features) and PHASE_LABELS
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
