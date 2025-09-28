from robojudo.config import ASSETS_DIR
from robojudo.controller.ctrl_cfgs import MotionCtrlCfg


class G1MotionCtrlCfg(MotionCtrlCfg):
    # ==== policy specific configs ====
    track_keypoints_names: list[str] = []
    phc: MotionCtrlCfg.PhcCfg = MotionCtrlCfg.PhcCfg(
        robot_config_file="robot/unitree_g1_29dof.yaml",
    )

    # ==== motion config ====
    robot: str = "g1"
    # PHC retargeted motion
    # motion_name: str = "dance_sample_g1"
    # motion_name: str = "singles/0-KIT_572_punch_right01_poses"
    motion_name: str = "singles/0-KIT_6_WalkInCounterClockwiseCircle05_1_poses"
    # motion_name: str = "singles/0-Transitions_mocap_mazen_c3d_dance_stand_poses"
    # motion_name: str = "amass_all_phc"

    @property
    def motion_path(self) -> str:
        motion_path = ASSETS_DIR / f"motions/{self.robot}/phc_29/{self.motion_name}.pkl"
        return motion_path.as_posix()
