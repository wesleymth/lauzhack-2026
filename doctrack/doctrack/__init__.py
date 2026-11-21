"""DocTrack: find documents on a desk and track their 4 corners in real time."""

from .detector import Detection, Detector, OpenCVDocumentDetector
from .geometry import (estimate_aspect, order_corners, paper_size, point_in_quad, quad_iou, refine_quad,
                       warp_document)
from .locked import LockedTracker
from .tracker import DocumentTracker, Track

__all__ = [
    "Detection",
    "Detector",
    "OpenCVDocumentDetector",
    "DocumentTracker",
    "Track",
    "order_corners",
    "point_in_quad",
    "quad_iou",
    "warp_document",
    "refine_quad",
    "estimate_aspect",
    "paper_size",
    "LockedTracker",
]
