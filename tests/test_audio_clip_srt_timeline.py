import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "funclip"))

from videoclipper import VideoClipper


def _state():
    sentences = [
        {"text": "hello", "timestamp": [[1000, 2000]], "spk": 0},
        {"text": "world", "timestamp": [[4000, 5000]], "spk": 0},
    ]
    return {
        "audio_input": (16000, np.zeros(96000)),
        "recog_res_raw": "hello world",
        "timestamp": [[1000, 2000], [4000, 5000]],
        "sentences": sentences,
        "sd_sentences": copy.deepcopy(sentences),
    }


def test_concatenated_audio_subtitles_accumulate_output_time():
    clipper = VideoClipper(None)

    (rate, audio), _, subtitles = clipper.clip(
        "hello#world", 0, 0, copy.deepcopy(_state())
    )

    assert rate == 16000
    assert len(audio) == 32000
    assert "00:00:00,000 --> 00:00:01,000" in subtitles
    assert "00:00:01,000 --> 00:00:02,000" in subtitles


def test_explicit_timestamp_list_uses_same_accumulated_timeline():
    clipper = VideoClipper(None)

    (_, audio), _, subtitles = clipper.clip(
        "", 0, 0, copy.deepcopy(_state()),
        timestamp_list=[[16000, 32000], [64000, 80000]],
    )

    assert len(audio) == 32000
    assert "00:00:01,000 --> 00:00:02,000" in subtitles
