import os
import time

import numpy as np

from robojudo.config.g1.env.g1_dummy_env_cfg import G1DummyEnvCfg
from robojudo.environment.dummy_env import DummyEnv
from robojudo.tools.tool_cfgs import ForwardKinematicCfg


def get_latest_folder(path, index=-1):
    folder_list = os.listdir(path)
    folder_list = [os.path.join(path, f) for f in folder_list]
    folder_list = list(filter(lambda f: os.path.isdir(f), folder_list))
    folder_list.sort(key=lambda f: os.path.getmtime(f))
    return folder_list[index]


def view_log(folder_path):
    log_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npz")])
    if not log_files:
        exit()

    fk_cfg = ForwardKinematicCfg(
        xml_path=G1DummyEnvCfg.model_fields["xml"].default,
        debug_viz=True,
    )
    env_cfg = G1DummyEnvCfg(forward_kinematic=fk_cfg, odometry_type="DUMMY")

    env = DummyEnv(cfg_env=env_cfg)

    for _i, log_file in enumerate(log_files):
        file_path = os.path.join(folder_path, log_file)
        log_frame = np.load(file_path, allow_pickle=True)

        env_data = log_frame["env_data"][()]
        ctrl_data = log_frame["ctrl_data"][()]
        extras = log_frame["extras"][()]
        pd_target = log_frame["pd_target"]
        timestep = log_frame["timestep"]
        time_then = log_frame["time"]

        joint_pos = env_data["dof_pos"]
        base_pos = env_data["base_pos"]
        base_quat = env_data["base_quat"]
        env.kinematics.forward(
            joint_pos=joint_pos,
            base_pos=base_pos,
            base_quat=base_quat,
        )
        print(f"Step {timestep}")
        time.sleep(0.005)


def plot_log(folder_path):
    log_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npz")])
    log_files = log_files[1000:1500]
    if not log_files:
        exit()
    plot_data = [
        [0],
        [],
    ]
    for _i, log_file in enumerate(log_files):
        file_path = os.path.join(folder_path, log_file)
        log_frame = np.load(file_path, allow_pickle=True)

        env_data = log_frame["env_data"][()]
        ctrl_data = log_frame["ctrl_data"][()]
        extras = log_frame["extras"][()]
        pd_target = log_frame["pd_target"]
        timestep = log_frame["timestep"]
        time_then = log_frame["time"]

        joint_pos = env_data["dof_pos"]
        base_pos = env_data["base_pos"]
        base_quat = env_data["base_quat"]

        plot_data[0].append(joint_pos[13])  # waist roll
        plot_data[1].append(pd_target[13])

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(plot_data[0], color="blue", label="DOF Position")
    ax1.set_xlabel("Index", fontsize=12)
    ax1.set_ylabel("DOF Position", color="blue", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.plot(plot_data[1], color="red", label="PD TARGET")
    ax2.set_ylabel("PD target", color="red", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="red")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9), fontsize=10)

    plt.title("DOF Position", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    folder_path = get_latest_folder("logs/logs-g1-2-0924sii.tar", 0)
    print(folder_path)
    plot_log(folder_path)
    # view_log(folder_path)
