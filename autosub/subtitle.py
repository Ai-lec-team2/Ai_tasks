from faster_whisper import WhisperModel # faster-whisper
import os
import sys

#https://medium.com/@johnidouglasmarangon/build-a-speech-to-text-service-in-python-with-faster-whisper-39ad3b1e2305

audio = os.path.join(os.path.dirname(__file__), "test.mp3")

model_size = "small"  # https://github.com/SYSTRAN/faster-whisper?tab=readme-ov-file#model-conversion
model = WhisperModel(model_size, device="cpu", compute_type="float32")

segments, info = model.transcribe(audio, language="en", vad_filter=True, word_timestamps=True)
transcript = [{
                "start": s.start,
                "end": s.end,
                "text": s.text,
            }for s in segments]

print(transcript)