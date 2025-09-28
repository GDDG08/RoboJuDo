from typing import Literal

from robojudo.environment.env_cfgs import UnitreeEnvCfg

# from robojudo.tools.tool_cfgs import ZedOdometryCfg
from .g1_env_cfg import G1EnvCfg


class G1RealEnvCfg(G1EnvCfg, UnitreeEnvCfg):
    env_type: str = UnitreeEnvCfg.model_fields["env_type"].default
    # ====== ENV CONFIGURATION ======
    unitree: UnitreeEnvCfg.UnitreeCfg = UnitreeEnvCfg.UnitreeCfg(
        net_if="eth0",
        # net_if = "enp13s0",
        robot="g1",
        msg_type="hg",
        hand_type="NONE",
        lowcmd_topic="rt/lowcmd",
        lowstate_topic="rt/lowstate",
        enable_odometry=True,
    )

    odometry_type: Literal["DUMMY", "UNITREE", "ZED"] = "UNITREE"
    # zed_cfg: ZedOdometryCfg | None = ZedOdometryCfg(
    #     server_ip="192.168.123.167",
    #     pos_offset=[0.0, 0.0, 0.9],
    #     zero_align=True,
    # )

    joint2motor_idx: list[int] | None = None  # list(range(0, 29))


class G1WithHandRealEnvCfg(G1EnvCfg, UnitreeEnvCfg):
    env_type: str = UnitreeEnvCfg.model_fields["env_type"].default
    # ====== ENV CONFIGURATION ======
    unitree: UnitreeEnvCfg.UnitreeCfg = UnitreeEnvCfg.UnitreeCfg(
        net_if="eth0",
        # net_if = "enp13s0",
        robot="g1",
        msg_type="hg",
        hand_type="Dex-3",
        lowcmd_topic="rt/lowcmd",
        lowstate_topic="rt/lowstate",
        enable_odometry=True,
    )

    odometry_type: Literal["DUMMY", "UNITREE", "ZED"] = "UNITREE"
    # zed_cfg: ZedOdometryCfg | None = ZedOdometryCfg(
    #     server_ip="192.168.123.167",
    #     pos_offset=[0.0, 0.0, 0.9],
    #     zero_align=True,
    # )

    joint2motor_idx: list[int] | None = None  # list(range(0, 29))
