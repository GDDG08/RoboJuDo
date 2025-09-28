# from .motion_lib import MotionLibReal
import logging
import os
import time
from queue import Queue
from threading import Lock, Thread, Timer

import numpy as np
import torch
from box import Box
from phc.utils.motion_lib_base import FixHeightMode
from phc.utils.motion_lib_real import MotionLibReal
from poselib.poselib.skeleton.skeleton3d import SkeletonTree

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import MotionCtrlCfg
from robojudo.controller.motion_gui import MotionGUI
from robojudo.environment import Environment
from robojudo.utils.util_func import my_quat_rotate_np, to_torch

logger = logging.getLogger(__name__)


# TODO: motion done flag
@ctrl_registry.register
class MotionCtrl(Controller):
    cfg_ctrl: MotionCtrlCfg
    env: Environment

    def __init__(
        self,
        cfg_ctrl,
        env,
        device="cpu",
    ):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        assert self.env is not None, "Env is required for MotionCtrl"
        self.enable_gui = self.cfg_ctrl.motion_ctrl_gui
        self.motion_path = self.cfg_ctrl.motion_path
        self.extra_motion_data = self.cfg_ctrl.extra_motion_data

        assert os.path.exists(self.motion_path), f"Motion file {self.motion_path} not found!"

        # ========== PHC config ==========
        phc_robot_config = self.cfg_ctrl.phc.robot_config
        phc_robot_config["has_upright_start"] = False

        # ========== motion body keypoints ==========
        extend_config = phc_robot_config["extend_config"]

        self.motion_body_names = phc_robot_config["body_names"]
        motion_track_bodies_id = [
            self.motion_body_names.index(body_name) for body_name in self.cfg_ctrl.track_keypoints_names
        ]
        self.motion_track_bodies_extend_id = motion_track_bodies_id + list(
            range(len(self.motion_body_names), len(self.motion_body_names) + len(extend_config))
        )

        # ========== robot body keypoints ==========
        assert self.env.kinematics is not None, "Env Kinematics model is required for MotionCtrl"
        robot_body_names = self.env.kinematics.body_names

        robot_track_bodies_id = [robot_body_names.index(body_name) for body_name in self.cfg_ctrl.track_keypoints_names]

        self.extend_body_parent_names = []
        self.extend_body_parent_ids = []
        extend_body_pos_list = []
        for cfg in extend_config:
            parent_name, pos = cfg["parent_name"], cfg["pos"]
            extend_body_parent_id = robot_body_names.index(parent_name)
            self.extend_body_parent_names.append(parent_name)
            self.extend_body_parent_ids.append(extend_body_parent_id)
            extend_body_pos_list.append(pos)
        self.extend_body_pos = np.asarray(extend_body_pos_list).reshape(-1, 3)

        self.robot_track_bodies_extend_id = robot_track_bodies_id + list(
            range(len(robot_body_names), len(robot_body_names) + len(extend_config))
        )

        # ========== PHC motionlib ==========
        asset_file = phc_robot_config["asset"]["assetFileName"]
        self.skeleton_tree = SkeletonTree.from_mjcf(asset_file)

        motion_lib_cfg = Box(
            {
                "motion_file": self.motion_path,
                # "fix_height": FixHeightMode.full_fix,
                "fix_height": FixHeightMode.no_fix,
                "min_length": -1,
                "max_length": -1,
                "im_eval": False,
                "multi_thread": False,
                "smpl_type": phc_robot_config["humanoid_type"],
                "randomrize_heading": False,
                "device": self.device,
                "robot": phc_robot_config,
            }
        )

        self._motion_lib = MotionLibReal(motion_lib_cfg)
        self.ref_motion_cache = {}

        # ========== Play Control ==========
        self.motion_time = 0
        self.motion_id = -1
        self.motion_name = "Blank Name"
        self.motion_length = 0.01
        self.motion_offset = np.array([0.0, 0.0, 0.0])
        self.motion_target_heading = np.array([0.0, 0.0, 0.0, 1.0])  # Default target heading

        self.play_speed_ratio = 1

        self.fade_step = 0.01
        self.fade_delay = 20  # ms
        self.speed_target = 1 / 2
        self.speed_steps = iter(np.arange(0, self.speed_target + self.fade_step, self.fade_step))

        if self.enable_gui:
            self.motion_gui = MotionGUI(self)
            self.play_speed_ratio = 0

        self.lock_motion_load = Lock()
        self.gui_commands = Queue()

        # init
        self.load_motion(0, block=True)

    def load_motion(self, motion_id=None, block=False):
        if (self.motion_id >= 0) and (motion_id == self.motion_id):
            logger.debug(f"[MotionCtrl] motion_id {motion_id} already loaded")
            return

        if self.lock_motion_load.locked():
            logger.debug(f"[MotionCtrl] motion_id {motion_id} already loading")
            return

        def load_motion_thread():
            motion_start_idx = motion_id if motion_id is not None else 0
            with self.lock_motion_load:
                self._motion_lib.load_motions(
                    skeleton_trees=[self.skeleton_tree],
                    gender_betas=[torch.zeros(17)],
                    limb_weights=[np.zeros(10)],
                    random_sample=False,
                    start_idx=motion_start_idx,
                    target_heading=self.motion_target_heading,
                )

                self.motion_id = motion_start_idx
                # self.motion_dt = self._motion_lib._motion_dt
                self.motion_name = self._motion_lib.curr_motion_keys
                self.motion_length = self._motion_lib._motion_lengths[0]
                self.reset()
                # TODO check auto reset

            if self.enable_gui:
                self.motion_gui.update_info(f"{self.motion_id}@{self.motion_name}", self.motion_length)

        if block:
            load_motion_thread()
        else:
            load_thread = Thread(target=load_motion_thread, daemon=True)
            load_thread.start()

    def get_motion(self):
        if self.lock_motion_load.locked():
            logger.debug("[MotionCtrl] use cache motion as loading lock")
            return self.ref_motion_cache
        # read motion
        # logger.debug(f"{self.motion_id=}, {self.motion_timestep=}")

        motion_ids = to_torch([0], dtype=torch.int32)
        motion_times = to_torch(self.motion_time).repeat(1)
        offset = to_torch(self.motion_offset).repeat(1)

        ## Cache the motion + offset
        if (
            offset is None
            or "motion_ids" not in self.ref_motion_cache
            or self.ref_motion_cache["offset"] is None
            or len(self.ref_motion_cache["offset"]) != len(offset)
            or (self.ref_motion_cache["motion_times"] - motion_times).abs().sum()
            + (self.ref_motion_cache["offset"] - offset).abs().sum()
            > 0
        ):
            self.ref_motion_cache["motion_ids"] = motion_ids.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache["motion_times"] = motion_times.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache["offset"] = offset.clone() if offset is not None else None
        else:
            return self.ref_motion_cache
        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=offset)

        self.ref_motion_cache.update(motion_res)

        return self.ref_motion_cache

    def get_robot_state(self):
        fk_info = self.env.fk_info
        assert fk_info is not None, "Env fk_info is required for MotionCtrl"

        body_pos = np.array([body_info["pos"] for body_info in fk_info.values()])
        body_rot = np.array([body_info["quat"] for body_info in fk_info.values()])

        extend_curr_pos = (
            my_quat_rotate_np(
                body_rot[self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos.reshape(-1, 3)
            ).reshape(-1, 3)
            + body_pos[self.extend_body_parent_ids]
        )
        body_pos_extend = np.concatenate([body_pos, extend_curr_pos], axis=0)
        body_pos_subset = body_pos_extend[self.robot_track_bodies_extend_id, :]

        return body_pos_extend, body_pos_subset

    def get_data(self):
        motion_res = self.get_motion()
        ref_body_pos_extend = motion_res["rg_pos_t"].cpu().numpy().squeeze().copy()
        ref_body_vel_extend = motion_res["body_vel_t"].cpu().numpy().squeeze().copy() * self.play_speed_ratio
        ref_body_pos_subset = ref_body_pos_extend[self.motion_track_bodies_extend_id]
        ref_body_vel_subset = ref_body_vel_extend[self.motion_track_bodies_extend_id]

        body_pos_extend, body_pos_subset = self.get_robot_state()

        ctrl_data = {
            "ref_body_pos_subset": ref_body_pos_subset,
            "ref_body_vel_subset": ref_body_vel_subset,
            "robot_body_pos_subset": body_pos_subset,
            "dof_pos": motion_res["dof_pos"].cpu().numpy().squeeze().copy(),
        }

        if (hand_pose := motion_res.get("hand_pose", None)) is not None:
            ctrl_data["hand_pose"] = hand_pose.cpu().numpy().squeeze().copy().reshape(2, -1)

        if self.extra_motion_data:
            ctrl_data.update(
                {
                    "_motion_track_bodies_extend_id": self.motion_track_bodies_extend_id,
                    "_robot_track_bodies_extend_id": self.robot_track_bodies_extend_id,
                    "rg_pos_t": ref_body_pos_extend,
                    "body_vel_t": ref_body_vel_extend,
                    # extra for motion recognition
                    "root_pos": motion_res["root_pos"].cpu().numpy().squeeze().copy(),
                    "root_rot": motion_res["root_rot"].cpu().numpy().squeeze().copy(),
                    "root_vel": motion_res["root_vel"].cpu().numpy().squeeze().copy() * self.play_speed_ratio,
                    "root_ang_vel": motion_res["root_ang_vel"].cpu().numpy().squeeze().copy() * self.play_speed_ratio,
                    "freq": motion_res["freq"].cpu().numpy().squeeze().copy(),
                    "phase": motion_res["phase"].cpu().numpy().squeeze().copy(),
                }
            )

        return ctrl_data

    def process_triggers(self, ctrl_data):
        commands = []
        while not self.gui_commands.empty():
            command = self.gui_commands.get()
            if command not in commands:
                commands.append(command)

        return ctrl_data, commands

    def post_step_callback(self, commands=None, motion_time_step=0.02):
        if commands is None:
            commands = []
        # logger.debug(f"{self.play_speed_ratio=}")
        self.motion_time += motion_time_step * self.play_speed_ratio

        # if (self.motion_time) > (self.motion_length + 1):
        #     # self.motion_timestep = 0
        #     self.load_motion(self.motion_id + 1)
        #     # self.motion_offset = -self._motion_lib.get_root_pos_smpl([0], to_torch([0]))["root_pos"][0]/
        #       + self.zed_odometry.get_status()["position_xyz"]
        #     # logger.debug(f"{self.motion_offset=}")
        #     self.reset()
        if self.enable_gui:
            self.motion_gui.update_time(self.motion_time)
        # pass

        for command in commands:
            match command:
                case "[MOTION_FADE_IN]":
                    self.fade_in()
                case "[MOTION_FADE_OUT]":
                    self.fade_out()
                case "[MOTION_RESET]":
                    self.reset()
                case "[MOTION_LOAD_NEXT]":
                    self.load_motion(self.motion_id + 1, block=False)
                case "[MOTION_LOAD_PREV]":
                    if self.motion_id > 0:
                        self.load_motion(self.motion_id - 1, block=False)

    # ========== Play Control ==============
    def reset(self):
        self.motion_time = 0

        motion_init_pos = self._motion_lib.get_root_pos_smpl([0], to_torch([0]))["root_pos"][0]
        motion_init_pos[2] = 0.0
        self.motion_offset = -motion_init_pos
        # logger.debug(f"{self.motion_offset=}")

    def _fade_step_apply(self):
        try:
            speed_step = float(next(self.speed_steps))
            speed_step = max(speed_step, 0)
            self.play_speed_ratio = speed_step

            Timer(self.fade_delay / 1000.0, self._fade_step_apply).start()
        except StopIteration:
            logger.info("Fade-step complete")

    def fade_in(self):
        self.speed_steps = iter(np.arange(self.play_speed_ratio, self.speed_target + self.fade_step, self.fade_step))
        self._fade_step_apply()

    def fade_out(self):
        self.speed_steps = iter(np.arange(self.play_speed_ratio, 0 - self.fade_step, -self.fade_step))
        self._fade_step_apply()


if __name__ == "__main__":
    from pprint import pprint

    from robojudo.config.g1.ctrl.g1_motion_ctrl_cfg import G1MotionCtrlCfg
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg
    from robojudo.environment.mujoco_env import MujocoEnv

    env = MujocoEnv(cfg_env=G1MujocoEnvCfg(xml="assets/robots/g1/xml/g1_29dof_rev_1_0.xml"))
    env.reset()
    motion_ctrl = MotionCtrl(
        cfg_ctrl=G1MotionCtrlCfg(
            motion_name="singles/0-Eyes_Japan_Dataset_hamada_gesture_etc-23-mobile call-hamada_poses.pkl"  # noqa: E501
        ),
        env=env,
    )

    while True:
        motion_ctrl.post_step_callback()
        motion_data = motion_ctrl.get_data()
        pprint(motion_data)
        motion_ctrl.post_step_callback()
        time.sleep(0.1)
