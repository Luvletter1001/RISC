from .base import BaseDetectorAdapter


class GroundingDINOAdapter(BaseDetectorAdapter):
    def load(self, device: str):
        raise FileNotFoundError("NOT_AVAILABLE: groundingdino adapter is a stub until inventory finds a supported entry")

