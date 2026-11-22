"""MobileSAM page outline at lock time (sam_select.py)."""
import numpy as np
import pytest

from bench.track_bench import corner_error, page_texture, quad_at, render
from doctrack import DocumentTracker, OpenCVDocumentDetector
from doctrack.sam_select import sam_available

pytestmark = pytest.mark.skipif(not sam_available(), reason="onnxruntime not available")
TEX = page_texture()
LIGHT_DESK = np.clip(np.full((1080, 1920, 3), 214, np.float32)
                     + np.random.default_rng(0).normal(0, 4, (1080, 1920, 1)), 0, 255).astype(np.uint8)


def _lock(use_sam):
    q = quad_at(900, 560, 8)
    img = render(LIGHT_DESK, TEX, q, rng=np.random.default_rng(1))
    tr = DocumentTracker(OpenCVDocumentDetector(), use_sam=use_sam)
    for i in range(14):
        tr.update(img, i / 30)
    return tr.select_at(*q.mean(0)), q


def test_sam_locks_a_low_contrast_page_precisely():
    t, q = _lock(use_sam=True)
    assert t is not None
    assert corner_error(t.corners, q) < 6.0          # then followed by the docaligner tracker


def test_sam_outline_is_precise():
    from doctrack.sam_select import PageSegmenter
    q = quad_at(900, 560, 8)
    img = render(LIGHT_DESK, TEX, q, rng=np.random.default_rng(1))
    out = PageSegmenter().page_quad(img, q.mean(0))
    assert out is not None and corner_error(out, q) < 1.5
