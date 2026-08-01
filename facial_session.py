# facial_session.py
import time

class FacialSession:
    def __init__(self, window_sec=10):
        self.window_sec = window_sec
        self.samples = []  # (timestamp, score)
        self.blink_timestamps = []

    def update(self, score: float, ear: float, blink_threshold=0.21):
        now = time.time()
        self.samples.append((now, score))
        self.samples = [(t, s) for t, s in self.samples if now - t <= self.window_sec]

        if ear < blink_threshold:
            self.blink_timestamps.append(now)
        self.blink_timestamps = [t for t in self.blink_timestamps if now - t <= 60]

    def current_score(self):
        if not self.samples:
            return 50.0  # neutral default if no face data yet
        return sum(s for _, s in self.samples) / len(self.samples)

    def blink_rate_per_min(self):
        return len(self.blink_timestamps)