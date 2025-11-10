# Environment

**Environment** is the component that interact with the robot. It receives the action from the policy and executes it on the robot.

## [Environment](#environment)
`Environment`. It is an abstract class that defines the interface for the environment. The environment can be either a real robot or a simulated robot.

`Environment` offer feedback properties. including:
- `dof_pos`: `np.ndarray`: the angle of the joints.
- `dof_vel`: `np.ndarray`: the velocity of the joints.
- `base_quat`: `np.ndarray`: the quaternion of the root. `w` last.
- `base_ang_vel`: `np.ndarray`: the angular velocity of the root.
- `base_lin_acc`: `np.ndarray`: the linear acceleration of the root.

**optional variables, may not be available in all envs according to config:**
- `base_pos`: `np.ndarray`: the 3D position of the root.
- `base_lin_vel`: `np.ndarray`: the linear velocity of the root.
- `torso_pos`: `np.ndarray`: the 3D position of the torso.
- `torso_quat`: `np.ndarray`: the quaternion of the torso.
- `torso_ang_vel`: `np.ndarray`: the angular velocity of the torso.
- `fk_info`: `dict`: the forward kinematics information:
  - `{body_name}`:
    - `pos`: `np.ndarray`: the 3D position of the each link.
    - `quat`: `np.ndarray`: the quaternion of the each link. `w` last.
    - `ang_vel`: `np.ndarray`: the angular velocity of the each link.
    - `lin_vel`: `np.ndarray`: the linear velocity of the each link.

### ✨ Feature: Dynamic DoF Config

The [`DoFConfig`](../robojudo/tools/tool_cfgs.py) in environments defines `joint_names`, `stiffness`, `default_pos`,etc. It supports:
* **Subset extension & cropping** — easily handle locked DoFs using `_subset`.
  * Example: [G1_12DoF](../robojudo/config/g1/g1_cfg.py).
* **Dynamic override by policy config** — policies can override environment DoF settings, even when `joint_names` are not aligned.
  * For instance, run a **12-DoF policy** on a **29-DoF environment**.
* **Strict checks** — all configs are validated in `DoFConfig` to ensure safety.

### ✨ Feature: Born Place Alignment

The robot’s **base quaternion** and **base position** can be easily aligned during deployment. 

This allows the robot’s starting pose to be corrected to the expected zero point, without adding complex world2init transformations inside the policy.

### ✨ Feature: Odometry Support

RoboJuDo supports odometry, enabling tracking of the robot’s global position and velocity. We have two options now:
- `ZED`: Use a ZED camera for odometry. `zed_proxy` submodule is needed.
- `UNITREE`: Use the built-in sport state service of Unitree robots.

This allows policies and controllers to leverage absolute positioning for more reliable deployment and motion alignment.

### ✨ Feature: Forward Kinematic
RoboJuDo provides simple access to rich information such as link poses, joint positions, and end-effector states. This makes it easy to implement advanced controllers and monitoring tools without additional computation overhead.

> 💡Tips: You can use the debug_viz option for real-time sim2real monitoring. Check [ForwardKinematicCfg](../robojudo/tools/tool_cfgs.py)

<img src="images/env-debug_viz.png" width="10%" alt="debug_viz"/>

---

Check [Environment](../robojudo/environment/base_env.py) and [EnvCfg](../robojudo/environment/env_cfgs.py) for more details.

---

We provide following Environments:
- [DummyEnv](#environment--dummy_env)
- [MujocoEnv](#environment--mujoco_env)
- [UnitreeEnv](#environment--unitree_env)
- [UnitreeCppEnv](#environment--unitreecppenv)

## [Environment](#environment) > [DummyEnv](#environment--dummyenv)

`DummyEnv` is the environment that does nothing. It is a subclass of `Environment` and implements the interface defined in `Environment`. It is mainly used for testing purposes. It will print some debug info when running.

script: [dummy_env.py](../robojudo/environment/dummy_env.py)

## [Environment](#environment) > [MujocoEnv](#environment--mujocoenv)

`MujocoEnv` is the environment that simulates the robot using Mujoco. It is a subclass of `Environment` and implements the interface defined in `Environment`.

script: [mujoco_env.py](../robojudo/environment/mujoco_env.py)

## [Environment](#environment) > [UnitreeEnv](#environment--unitreeenv)


`UnitreeEnv` is the environment that simulates the robot using Unitree's python SDK. It is a subclass of `Environment` and implements the interface defined in `Environment`.
> [`unitree_sdk2py`](https://github.com/unitreerobotics/unitree_sdk2_python) is needed.

script: [unitree_env.py](../robojudo/environment/unitree_env.py)


## [Environment](#environment) > [UnitreeCppEnv](#environment--unitreecppenv)

`UnitreeCppEnv` is the environment that simulates the robot using Unitree's C++ SDK. It is a subclass of `Environment` and implements the interface defined in `Environment`.

script: [unitree_cpp_env.py](../robojudo/environment/unitree_cpp_env.py)

> [UnitreeCpp](https://github.com/HansZ8/unitree_cpp) is needed.

This environment is implemented with [UnitreeCpp](https://github.com/HansZ8/unitree_cpp). In our test onboard Unitree G1, `UnitreeCppEnv` is faster almost 100 times than `UnitreeEnv` in `step()`.


