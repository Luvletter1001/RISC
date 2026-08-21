from .base import BaseDetectorAdapter


class YOLOWorldAdapter(BaseDetectorAdapter):
    def load(self, device: str):
        raise FileNotFoundError("NOT_AVAILABLE: yoloworld adapter is a stub until inventory finds a supported entry")

