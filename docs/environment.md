# Environment

**Environment** is the component that simulates the robot. It receives the action from the policy and executes it on the robot.

## [Environment](#environment)
`Environment`. It is an abstract class that defines the interface for the environment. The environment can be either a real robot or a simulated robot.

`Environment` offer `env_data` in `step` method. `env_data` is a dictionary that contains the sensor data of the robot. The sensor data includes the joint positions, velocities, accelerations, base pose, and base velocity:

`env_data`:
- `dof_pos`: `np.ndarray`: the angle of the joints.
- `dof_vel`: `np.ndarray`: the velocity of the joints.
- `base_quat`: `np.ndarray`: the quaternion of the root. `w` last.
- `base_ang_vel`: `np.ndarray`: the angular velocity of the root.
- `base_lin_acc`: `np.ndarray`: the linear acceleration of the root.
- `base_pos`: `np.ndarray`: the 3D position of the root.
- `base_lin_vel`: `np.ndarray`: the linear velocity of the root.
- `torso_pos`: `np.ndarray`: the 3D position of the torso.
- `torso_quat`: `np.ndarray`: the quaternion of the torso.
- `torso_ang_vel`: `np.ndarray`: the angular velocity of the torso.
- `fk_info`: `dict`: the forward kinematics information:
  - `pos`: `np.ndarray`: the 3D position of the each link.
  - `quat`: `np.ndarray`: the quaternion of the each link. `w` last.
  - `ang_vel`: `np.ndarray`: the angular velocity of the each link.
  - `lin_vel`: `np.ndarray`: the linear velocity of the each link.


we provide two environments: `MujocoEnv` and `UnitreeCppEnv`:
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

script: [unitree_env.py](../robojudo/environment/unitree_env.py)

## [Environment](#environment) > [UnitreeCppEnv](#environment--unitreecppenv)

`UnitreeCppEnv` is the environment that simulates the robot using Unitree's C++ SDK. It is a subclass of `Environment` and implements the interface defined in `Environment`.

script: [unitree_cpp_env.py](../robojudo/environment/unitree_cpp_env.py)

due to the limitation of Unitree's C++ SDK, you need to install our `unitree cpp` beforehead. In our test. `UnitreeCppEnv` is faster almost 100 times than `UnitreeEnv` in `step()`
