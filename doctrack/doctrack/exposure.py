"""Lock / unlock the webcam's auto-exposure (macOS, AVFoundation).

OpenCV's AVFoundation backend ignores CAP_PROP_EXPOSURE, and macOS offers no
manual exposure value for USB webcams: only "continuous auto" or "locked at the
current value". So: when the docs look right, lock; when the light changes, unlock.

The configuration lock is held while exposure is locked: on macOS the system
may otherwise put the device back to auto exposure.
"""

from typing import Optional

LOCKED, CONTINUOUS = 0, 2          # AVCaptureExposureMode values
VIDEO = "vide"                     # AVMediaTypeVideo


class ExposureLock:
    def __init__(self, name_hint: str = "LVNext"):
        self.device = None
        self.locked = False
        try:   # pyobjc-core is already installed (pyobjc-framework-vision): load the framework by hand
            import objc
            objc.loadBundle("AVFoundation", {}, bundle_path="/System/Library/Frameworks/AVFoundation.framework")
            Device = objc.lookUpClass("AVCaptureDevice")
        except Exception as e:
            print(f"exposure lock unavailable (macOS only): {e}")
            return
        devices = list(Device.devicesWithMediaType_(VIDEO))
        match = [d for d in devices if name_hint.lower() in str(d.localizedName()).lower()]
        dev: Optional[object] = match[0] if match else Device.defaultDeviceWithMediaType_(VIDEO)
        if dev is None or not dev.isExposureModeSupported_(LOCKED):
            print("exposure lock unavailable: camera does not support locking exposure")
            return
        self.device = dev

    @property
    def available(self) -> bool:
        return self.device is not None

    def toggle(self) -> None:
        self.set(not self.locked)

    def set(self, lock: bool) -> None:
        if self.device is None or lock == self.locked:
            return
        if lock:
            ok = self.device.lockForConfiguration_(None)
            if isinstance(ok, tuple):      # (ok, NSError) when pyobjc knows the out-parameter
                ok = ok[0]
            if not ok:
                print("exposure lock failed: camera busy (another app configuring it?)")
                return
            self.device.setExposureMode_(LOCKED)
        else:
            self.device.setExposureMode_(CONTINUOUS)
            self.device.unlockForConfiguration()
        self.locked = lock
        print("exposure " + ("LOCKED" if lock else "auto"), flush=True)

    def close(self) -> None:
        self.set(False)
