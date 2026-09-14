import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "funclip"))

from utils.trans_utils import pre_proc, proc


def test_proc_matches_contiguous_chinese_using_token_timestamps():
    raw = "嗯那么今天我们就简"
    timestamps = [
        [230, 290], [590, 650], [710, 770],
        [890, 950], [1010, 1070], [1250, 1310],
        [1430, 1490], [1610, 1670], [1970, 2030],
    ]

    assert proc(raw, timestamps, pre_proc(raw[:8])) == [[3680, 26720]]
    assert proc(raw, timestamps, pre_proc("简")) == [[31520, 32480]]


def test_proc_preserves_ascii_case_insensitive_matching():
    timestamps = [[0, 100], [100, 200]]
    assert proc("Hello WORLD", timestamps, "hello world") == [[0, 3200]]


def test_proc_returns_all_repeated_matches():
    timestamps = [[0, 100], [100, 200], [200, 300], [300, 400]]
    assert proc("哈哈哈哈", timestamps, pre_proc("哈哈")) == [
        [0, 3200], [1600, 4800], [3200, 6400]
    ]
