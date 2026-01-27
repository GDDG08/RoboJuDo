import logging
import threading
import time
from typing import Any

import numpy as np

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import GoalkeeperBallDdsCtrlCfg

logger = logging.getLogger(__name__)

try:
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.sub import DataReader, Subscriber
    from cyclonedds.topic import Topic

    from robojudo.dds.detection_module import DetectionResults

    _DDS_AVAILABLE = True
    _DDS_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - optional dependency
    DomainParticipant = None
    Subscriber = None
    Topic = None
    DataReader = None
    DetectionResults = None
    _DDS_AVAILABLE = False
    _DDS_IMPORT_ERROR = e


def pick_best_ball_detection(results: list[Any]):
    best = None
    best_score = -1.0
    for r in results:
        class_id = str(getattr(r, "class_id", ""))
        class_name = str(getattr(r, "class_name", "")).lower()
        if class_id != "0" and class_name != "ball":
            continue
        score = float(getattr(r, "score", -1.0))
        if score > best_score:
            best = r
            best_score = score
    if best is None:
        return None
    return {"xyz": np.asarray(getattr(best, "xyz", [0.0, 0.0, 0.0]), dtype=np.float32), "score": best_score}


@ctrl_registry.register
class GoalkeeperBallDdsCtrl(Controller):
    cfg_ctrl: GoalkeeperBallDdsCtrlCfg

    def __init__(self, cfg_ctrl: GoalkeeperBallDdsCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)

        self.ball_xyz_cam = np.zeros(3, dtype=np.float32)
        self.ball_score = -1.0
        self.ball_valid = False
        self.last_update_time = 0.0

        if not _DDS_AVAILABLE:
            raise RuntimeError(
                "cyclonedds is required for GoalkeeperBallDdsCtrl. "
                "Please install cyclonedds or ensure it is available in your runtime."
            ) from _DDS_IMPORT_ERROR

        self.participant = DomainParticipant(self.cfg_ctrl.domain_id)  # type: ignore[call-arg]
        self.subscriber = Subscriber(self.participant)  # type: ignore[call-arg]
        self.topic = Topic(self.participant, self.cfg_ctrl.topic_name, DetectionResults)  # type: ignore[arg-type]
        self.reader = DataReader(self.subscriber, self.topic)  # type: ignore[call-arg]

        self._stop = False
        self._thread = threading.Thread(target=self._dds_worker, daemon=True)
        self._thread.start()

    def _dds_worker(self):
        while not self._stop:
            try:
                samples = self.reader.take()  # type: ignore[union-attr]
            except Exception as e:
                logger.debug(f"[GoalkeeperBallDdsCtrl] DDS read error: {e}")
                time.sleep(self.cfg_ctrl.poll_interval)
                continue

            if samples:
                for sample in samples:
                    results = getattr(sample, "results", None)
                    if results is None:
                        continue
                    best = pick_best_ball_detection(results)
                    if best is None:
                        continue
                    self.ball_xyz_cam = best["xyz"]
                    self.ball_score = float(best["score"])
                    self.ball_valid = True
                    self.last_update_time = time.time()

            time.sleep(self.cfg_ctrl.poll_interval)

    def reset(self):
        self.ball_xyz_cam[:] = 0.0
        self.ball_score = -1.0
        self.ball_valid = False
        self.last_update_time = 0.0

    def get_data(self):
        if self.ball_valid and self.cfg_ctrl.timeout_s > 0:
            if time.time() - self.last_update_time > self.cfg_ctrl.timeout_s:
                self.ball_valid = False

        return {
            "ball_xyz_cam": self.ball_xyz_cam.copy(),
            "ball_score": float(self.ball_score),
            "ball_valid": bool(self.ball_valid),
            "last_update_time": float(self.last_update_time),
        }

    def close(self):
        self._stop = True
