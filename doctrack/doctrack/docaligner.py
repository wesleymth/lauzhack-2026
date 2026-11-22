"""DocAligner: pretrained document-corner model (heatmaps for the 4 corners).

https://github.com/docsaidlab/docaligner, used here directly through
onnxruntime (the pip package pulls in a system JPEG library we don't need).
The model expects the document to fill most of the image, so the tracker
feeds it a square crop around the page. Weights (~13 MB) are downloaded on
first use into doctrack/models/.
"""

import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS = {   # name -> Google Drive file id (from the DocAligner package)
    "lcnet100": "1IlbLPkCv-TdaBLOPh_4J97P1_KHYYJ7a",
    "fastvit_t8": "1W88oOkHjMza6xXbDxrG8k_C-Tf2aYKJ4",
    "fastvit_sa24": "14vUH77v6yGg7zFctUgcT6BzV5Iisg4Dl",
}


def model_path(name: str) -> Path:
    return MODELS_DIR / f"{name}_h_e_bifpn_256_fp32.onnx"


def ensure_model(name: str = "fastvit_t8") -> Path:
    path = model_path(name)
    if not path.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.usercontent.google.com/download?id={MODELS[name]}&export=download&confirm=t"
        print(f"downloading DocAligner model {name} to {path} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        if tmp.stat().st_size < 1_000_000:
            tmp.unlink()
            raise RuntimeError(f"could not download {name}; put {path.name} in {MODELS_DIR} manually")
        tmp.rename(path)
    return path


def docaligner_available(name: str = "fastvit_t8") -> bool:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


class DocAlignerModel:
    """Square image (the page filling most of it) -> (corners (4, 2) px, confidence).

    Confidence = the weakest of the 4 corner heatmap peaks (0..1)."""
    min_conf, strict_conf = 0.4, 0.6      # confidence scale of this model
    full_frame = False                    # needs the page to fill most of the image
    square_crops = True                   # its input is square: don't squeeze the page
    trust_under_hand = True               # heatmaps estimate covered corners

    def __init__(self, name: str = "fastvit_t8") -> None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(ensure_model(name)), so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        h, w = img_bgr.shape[:2]
        x = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
        x = np.transpose(x, (2, 0, 1)).astype(np.float32)[None] / 255.0
        hm = self.session.run(None, {self.input_name: x})[0][0]        # (4, 128, 128)
        pts, confs = [], []
        for c in hm[:4]:
            Hh, Wh = c.shape
            yy, xx = np.unravel_index(int(np.argmax(c)), c.shape)
            y0, y1, x0, x1 = max(0, yy - 2), min(Hh, yy + 3), max(0, xx - 2), min(Wh, xx + 3)
            win = np.clip(c[y0:y1, x0:x1], 0, None)
            gy, gx = np.mgrid[y0:y1, x0:x1]
            s = float(win.sum()) + 1e-9
            # heatmap cell centres -> image pixels
            pts.append([((gx * win).sum() / s + 0.5) * w / Wh, ((gy * win).sum() / s + 0.5) * h / Hh])
            confs.append(float(c.max()))
        return np.float32(pts), float(min(confs))
