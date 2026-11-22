"""Tests for the content-following (optical flow + homography) locked tracker."""
import numpy as np
import pytest

from bench.track_bench import desk, page_texture, run
from doctrack import DocumentTracker, OpenCVDocumentDetector
from doctrack.flow_track import FlowLockedTracker

TEX, BG = page_texture(), desk()


@pytest.mark.parametrize("scenario", ["still", "slide", "corner", "drag", "away"])
def test_follows_the_page(scenario):
    r = run("flow", scenario, TEX, BG)
    assert r is not None, "could not lock the page"
    assert r["mean"] < 3.0 and r["off"] < 0.05, r


def test_blank_page_hands_over_to_the_fallback():
    from bench.track_bench import quad_at, render
    blank = np.full_like(TEX, 238)
    q = quad_at(900, 560, 8)
    frame = render(BG, blank, q)
    called = []

    class Stub:
        corners, state, visible, scan_size = q, "tracking", np.ones(4, bool), (700, 990)

        def update(self, *a, **k):
            called.append(1)

    tr = FlowLockedTracker(frame, q, 0.0, fallback=lambda: Stub())
    assert tr.uses_fallback
    tr.update(frame, 0.03)
    assert called


def test_auto_backend_is_flow_on_this_branch():
    assert DocumentTracker(OpenCVDocumentDetector()).lock_backend == "flow"
