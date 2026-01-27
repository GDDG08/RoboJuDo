import numpy as np

from robojudo.controller.goalkeeper_ball_dds_ctrl import pick_best_ball_detection


class _DummyResult:
    def __init__(self, class_id, class_name, score, xyz):
        self.class_id = class_id
        self.class_name = class_name
        self.score = score
        self.xyz = xyz


def test_pick_best_ball_detection():
    results = [
        _DummyResult("1", "Goalpost", 0.9, [1, 2, 3]),
        _DummyResult("0", "Ball", 0.4, [0.1, 0.2, 0.3]),
        _DummyResult("0", "Ball", 0.8, [0.4, 0.5, 0.6]),
    ]
    best = pick_best_ball_detection(results)
    assert best is not None
    assert np.allclose(best["xyz"], np.array([0.4, 0.5, 0.6], dtype=np.float32))
