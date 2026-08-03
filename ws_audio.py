# # ws_audio.py
# import torch
# import numpy as np
# from faster_whisper import WhisperModel

# # stt_model = WhisperModel("large-v3", device="cpu", compute_type="int8")
# stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")
# # Loads once at startup (cached by torch.hub after first download)
# vad_model, vad_utils = torch.hub.load(
#     repo_or_dir="snakers4/silero-vad",
#     model="silero_vad",
#     force_reload=False,
#     onnx=False,
# )
# (get_speech_timestamps, _, _, VADIterator, _) = vad_utils

# SAMPLE_RATE = 16000
# FRAME_SAMPLES = 512          # Silero expects 512-sample chunks at 16kHz
# SILENCE_END_MS = 800
# MIN_SPEECH_MS = 300

# class AudioSession:
#     def __init__(self):
#         self.vad_iterator = VADIterator(vad_model, sampling_rate=SAMPLE_RATE)
#         self.buffer = bytearray()
#         self._accum = bytearray()
#         self.has_speech = False
#         self.silence_ms = 0
#         self.speech_ms = 0

#     def feed(self, chunk: bytes):
#         """Feed raw PCM16 bytes, return finalized utterance bytes if answer ended."""
#         self.buffer.extend(chunk)
#         finalized = None
#         frame_bytes = FRAME_SAMPLES * 2  # 16-bit samples

#         while len(self.buffer) >= frame_bytes:
#             frame = bytes(self.buffer[:frame_bytes])
#             del self.buffer[:frame_bytes]

#             audio_int16 = np.frombuffer(frame, dtype=np.int16)
#             audio_float32 = audio_int16.astype(np.float32) / 32768.0
#             audio_tensor = torch.from_numpy(audio_float32)

#             speech_dict = self.vad_iterator(audio_tensor, return_seconds=False)
#             is_speech = speech_dict is not None and "start" in speech_dict if speech_dict else False

#             # VADIterator returns dict on speech start/end events; for simpler
#             # per-frame speech/silence, use the model directly instead:
#             speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()
#             frame_is_speech = speech_prob > 0.5

#             frame_ms = int(FRAME_SAMPLES / SAMPLE_RATE * 1000)  # ~32ms

#             if frame_is_speech:
#                 self.speech_ms += frame_ms
#                 self.silence_ms = 0
#                 self.has_speech = True
#                 self._accum.extend(frame)
#             else:
#                 self.silence_ms += frame_ms
#                 if self.has_speech:
#                     self._accum.extend(frame)

#             if self.has_speech and self.silence_ms >= SILENCE_END_MS and self.speech_ms >= MIN_SPEECH_MS:
#                 finalized = bytes(self._accum)
#                 self._accum = bytearray()
#                 self.speech_ms = 0
#                 self.silence_ms = 0
#                 self.has_speech = False
#                 self.vad_iterator.reset_states()

#         return finalized


# def pcm_to_wav_bytes(pcm: bytes) -> bytes:
#     import io, wave
#     buf = io.BytesIO()
#     with wave.open(buf, "wb") as wf:
#         wf.setnchannels(1)
#         wf.setsampwidth(2)
#         wf.setframerate(SAMPLE_RATE)
#         wf.writeframes(pcm)
#     return buf.getvalue()


# def transcribe_pcm(pcm: bytes) -> str:
#     wav_bytes = pcm_to_wav_bytes(pcm)
#     import tempfile
#     with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
#         tmp.write(wav_bytes)
#         tmp.flush()
#         segments, _ = stt_model.transcribe(
#             tmp.name,
#             beam_size=10,
#             # language="en",
#             vad_filter=True,
#             vad_parameters=dict(min_silence_duration_ms=500),
# )
#         return " ".join(s.text for s in segments).strip()


# ws_audio.py
import webrtcvad
from faster_whisper import WhisperModel

# Small model = fits comfortably within tight memory limits (int8 quantized)
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")

vad = webrtcvad.Vad(2)  # aggressiveness 0-3 (higher = more aggressive filtering of non-speech)

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 16-bit mono PCM
SILENCE_END_MS = 800          # silence duration that marks "answer finished"
MIN_SPEECH_MS = 300           # ignore tiny noises/coughs


class AudioSession:
    def __init__(self):
        self.buffer = bytearray()
        self._accum = bytearray()
        self.speech_ms = 0
        self.silence_ms = 0
        self.has_speech = False

    def feed(self, chunk: bytes):
        """Feed raw PCM16 bytes, return finalized utterance bytes if answer ended."""
        self.buffer.extend(chunk)
        finalized = None

        while len(self.buffer) >= FRAME_BYTES:
            frame = bytes(self.buffer[:FRAME_BYTES])
            del self.buffer[:FRAME_BYTES]

            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if is_speech:
                self.speech_ms += FRAME_MS
                self.silence_ms = 0
                self.has_speech = True
                self._accum.extend(frame)
            else:
                self.silence_ms += FRAME_MS
                if self.has_speech:
                    self._accum.extend(frame)  # keep trailing silence for natural cutoff

            if self.has_speech and self.silence_ms >= SILENCE_END_MS and self.speech_ms >= MIN_SPEECH_MS:
                finalized = bytes(self._accum)
                self._accum = bytearray()
                self.speech_ms = 0
                self.silence_ms = 0
                self.has_speech = False

        return finalized


def pcm_to_wav_bytes(pcm: bytes) -> bytes:
    import io, wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


def transcribe_pcm(pcm: bytes) -> str:
    wav_bytes = pcm_to_wav_bytes(pcm)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(wav_bytes)
        tmp.flush()
        segments, _ = stt_model.transcribe(
            tmp.name,
            beam_size=10,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        return " ".join(s.text for s in segments).strip()