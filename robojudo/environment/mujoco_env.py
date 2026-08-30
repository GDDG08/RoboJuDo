import logging
import time

import mujoco
import mujoco_viewer
import numpy as np

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.utils.util_func import quat_rotate_inverse_np, quatToEuler

logger = logging.getLogger(__name__)


@env_registry.register
class MujocoEnv(Environment):
    cfg_env: MujocoEnvCfg

    def __init__(self, cfg_env: MujocoEnvCfg, device="cpu"):
        super().__init__(cfg_env=cfg_env, device=device)

        self.sim_duration = cfg_env.sim_duration
        self.sim_dt = cfg_env.sim_dt
        self.sim_decimation = cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation

        self.model = mujoco.MjModel.from_xml_path(cfg_env.xml)  # pyright: ignore[reportAttributeAccessIssue]
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]

        # Zero passive MJCF joint stiffness/damping (mjModel.jnt_stiffness /
        # dof_damping). Some source MJCFs (e.g. assets/robots/h1_2, copied
        # verbatim from ProtoMotions' own asset) bake in nonzero per-joint
        # <joint stiffness=... damping=...> spring-damper attributes that
        # ProtoMotions' own MujocoSimulator explicitly zeros at load time
        # (protomotions/simulator/mujoco/simulator.py: "_zero_passive_forces
        # -- we manage PD control ourselves, so passive forces would
        # double-count"). RoboJuDo always drives DOFs via its own PD (either
        # explicit torque-per-substep or the implicit position-actuator path
        # below) so leaving passive joint stiffness/damping nonzero silently
        # adds an uncommanded spring pulling every joint toward the MJCF's
        # neutral qpos0, fighting the policy. Found via the H1_2 "falls a
        # few seconds into held-out motion tracking" parity investigation
        # (2026-07-02) -- see robojudo_prep.md. No-op for assets (e.g. G1's)
        # that already have zero passive joint stiffness/damping.
        self.model.jnt_stiffness[:] = 0.0
        self.model.dof_damping[:] = 0.0

        self.use_implicit_pd = cfg_env.use_implicit_pd
        if self.use_implicit_pd:
            self._configure_implicit_pd_actuators(cfg_env)

        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        self.viewer = mujoco_viewer.MujocoViewer(
            self.model,
            self.data,
            width=1200,
            height=900,
            hide_menus=True,
        )
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -10.0
        self.viewer.cam.azimuth = 180.0
        # self.viewer._paused = True

        if cfg_env.visualize_extras:
            self.visualizer = MujocoVisualizer(self.viewer)
        else:
            self.visualizer = None

        self.last_time = time.time()
        self.random_heading = cfg_env.random_heading

        self._apply_random_heading()

        self.update()  # get initial state

    def _configure_implicit_pd_actuators(self, cfg_env: MujocoEnvCfg):
        """Convert raw <motor> torque actuators into MuJoCo native position
        (implicit PD) actuators, matching
        protomotions/simulator/mujoco/simulator.py's
        `_configure_actuators_for_pd()` (used when ProtoMotions'
        `MujocoSimulatorConfig.use_implicit_pd=True`, the DEFAULT for
        BUILT_IN_PD checkpoints -- see robojudo_prep.md "epoch_3400 parity
        gap" for how this was discovered: RoboJuDo previously always used
        *explicit* PD -- torque computed in Python each substep and written
        to raw torque actuators -- which is ProtoMotions'
        `use_implicit_pd=False` mode, not what most checkpoints are
        trained/evaluated under).

        force = gainprm[0] * ctrl + biasprm[0] + biasprm[1]*q + biasprm[2]*qd
              = kp * ctrl - kp * q - kd * qd = kp*(ctrl - q) - kd*qd

        After this, `data.ctrl[i]` holds the target POSITION for actuator i
        (not torque) -- MuJoCo computes and clips the PD force internally at
        every physics substep.
        """
        joint_names = cfg_env.dof.joint_names
        stiffness = np.asarray(cfg_env.dof.stiffness, dtype=np.float64)
        damping = np.asarray(cfg_env.dof.damping, dtype=np.float64)
        torque_limits = np.asarray(cfg_env.dof.torque_limits, dtype=np.float64)
        name_to_idx = {name: i for i, name in enumerate(joint_names)}

        for act_idx in range(self.model.nu):
            jnt_id = self.model.actuator_trnid[act_idx, 0]
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if jnt_name not in name_to_idx:
                logger.warning(f"[MujocoEnv] implicit PD: actuator joint '{jnt_name}' not in dof config, skipping")
                continue
            i = name_to_idx[jnt_name]
            kp = float(stiffness[i])
            kd = float(damping[i])
            effort = float(torque_limits[i])

            self.model.actuator_gainprm[act_idx, 0] = kp
            self.model.actuator_biastype[act_idx] = 1  # mjBIAS_AFFINE
            self.model.actuator_biasprm[act_idx, 0] = 0.0
            self.model.actuator_biasprm[act_idx, 1] = -kp
            self.model.actuator_biasprm[act_idx, 2] = -kd

            self.model.actuator_forcerange[act_idx, 0] = -effort
            self.model.actuator_forcerange[act_idx, 1] = effort
            self.model.actuator_ctrllimited[act_idx] = 0
            self.model.actuator_forcelimited[act_idx] = 1

        logger.info(f"[MujocoEnv] configured {self.model.nu} actuators as implicit-PD position actuators")

    def _apply_random_heading(self):
        """Rotate the root body by a random yaw if random_heading is enabled."""
        if not self.random_heading:
            return
        yaw = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(yaw / 2), np.sin(yaw / 2)
        q = self.data.qpos[3:7].copy()  # MuJoCo [w, x, y, z]
        # Pre-multiply by yaw rotation q_yaw=[c,0,0,s]: q_new = q_yaw ⊗ q
        self.data.qpos[3] = c * q[0] - s * q[3]
        self.data.qpos[4] = c * q[1] - s * q[2]
        self.data.qpos[5] = c * q[2] + s * q[1]
        self.data.qpos[6] = c * q[3] + s * q[0]

    def reborn(self, init_qpos=None):
        if init_qpos is not None:
            self.data.qpos[0:7] = init_qpos
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
            self._apply_random_heading()
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)

    def self_check(self):
        pass

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    def update(self, simple=False):  # TODO: clean sensors in xml
        """simple: only update dof pos & vel"""
        dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs :]
        dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs :]

        self._dof_pos = dof_pos.copy()
        self._dof_vel = dof_vel.copy()

        if simple:
            return

        quat = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
        ang_vel = self.data.qvel.astype(np.float32)[3:6]
        base_pos = self.data.qpos.astype(np.float32)[:3]
        lin_vel = self.data.qvel.astype(np.float32)[0:3]

        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)

        lin_vel = quat_rotate_inverse_np(quat, lin_vel)
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()

        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()

        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        if hand_pose is not None:
            logger.info("Hand pose-->", hand_pose)

        self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
        if self.viewer.is_alive:
            self.viewer.render()

        if self.use_implicit_pd:
            # Implicit PD: write position targets once; MuJoCo's actuator
            # model (configured in _configure_implicit_pd_actuators)
            # computes and clips PD force internally at every substep --
            # matches ProtoMotions MujocoSimulator's use_implicit_pd=True.
            self.data.ctrl = pd_target
            for _ in range(self.sim_decimation):
                mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
                self.update(simple=True)
        else:
            for _ in range(self.sim_decimation):
                torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)

                self.data.ctrl = torque

                mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
                self.update(simple=True)
        self.update(simple=False)

    def shutdown(self):
        self.viewer.close()


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)
