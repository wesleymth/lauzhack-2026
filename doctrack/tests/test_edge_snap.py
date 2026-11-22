"""Edge snapping of the DocAligner tracker's corners (synthetic bench)."""
import pytest

from bench.track_bench import desk, page_texture, run
from doctrack.docaligner import docaligner_available

pytestmark = pytest.mark.skipif(not docaligner_available(), reason="onnxruntime not available")
TEX, BG = page_texture(), desk()


@pytest.mark.parametrize("scenario,limit", [("still", 1.5), ("corner", 1.5), ("away", 1.5),
                                            ("slide", 4.0), ("drag", 4.0)])
def test_corners_on_the_page_edges(scenario, limit):
    r = run("docaligner", scenario, TEX, BG)
    assert r is not None and r["mean"] < limit and r["off"] < 0.05, r
