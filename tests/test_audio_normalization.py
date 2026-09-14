import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "funclip"))

from videoclipper import VideoClipper


class CaptureASR:
    def generate(self, data, **kwargs):
        self.samples = len(data)
        return [{
            "text": "test",
            "raw_text": "test",
            "timestamp": [[0, 500]],
            "sentence_info": [],
        }]


def test_recog_normalizes_stereo_audio_before_resampling():
    rate = 48000
    t = np.arange(rate) / rate
    audio = np.stack(
        [
            0.2 * np.sin(2 * np.pi * 440 * t),
            0.3 * np.sin(2 * np.pi * 880 * t),
        ],
        axis=1,
    )
    original = audio.copy()
    model = CaptureASR()
    clipper = VideoClipper(model)
    clipper.lang = "en"

    _, _, state = clipper.recog((rate, audio))

    stored_rate, stored_audio = state["audio_input"]
    assert model.samples == 16000
    assert stored_rate == 16000
    assert stored_audio.ndim == 1
    assert len(stored_audio) == 16000
    np.testing.assert_array_equal(audio, original)

    (out_rate, out), _, _ = clipper.clip(
        "", 0, 0, state, timestamp_list=[[0, 8000]]
    )
    assert out_rate == 16000
    assert len(out) == 8000
    assert len(out) / out_rate == 0.5
