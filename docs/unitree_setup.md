<h1 align="center"><b>Setup for Unitree Robots  </b></h1>

RoboJuDo has two environments for unitree robots:

- **`UnitreeEnv`**: Based on [`unitree_sdk2py`](https://github.com/unitreerobotics/unitree_sdk2_python).
    - Support both  `UnitreeH1` and `UnitreeG1`.
    - May endure performance issues on `UnitreeG1` due to limited computing power.
- **`UnitreeCppEnv`**: Based on [UnitreeCpp](https://github.com/HansZ8/unitree_cpp)
    - Support `UnitreeG1`.
    - Can be deployed onboard `UnitreeG1` pc2. It is much faster and stable.

# 🛠️SDK Installation

Choose the above Env&SDK options you want to use.

## UnitreeEnv

1. Follow the instuction to install the official SDK: [`unitree_sdk2py`](https://github.com/unitreerobotics/unitree_sdk2_python)
2. verify with 
    ```bash
    python -c "from robojudo.environment import UnitreeEnv"
    ```
    if any error, check installation of `unitree_sdk2py`

## UnitreeCppEnv

As the setup of [UnitreeCpp](https://github.com/HansZ8/unitree_cpp#installation) :

1. Install the Unitree official C++ SDK: [`unitree_sdk2`](https://github.com/unitreerobotics/unitree_sdk2)
    > Note: It is recommended to use the default installation path. 
2. and then install the `unitree_cpp` package:
    For simple setup, `unitree_cpp` is included in RoboJuDo, just run:
    ```bash
    python submodule_install.py unitree_cpp
    ```
    > ⭐ make sure `unitree_sdk2` is installed first
3. verify with 
    ```bash
    python -c "from robojudo.environment import UnitreeCppEnv"
    ```
    if any error, check installation of `unitree_sdk2` and then `unitree_cpp`.


# 🤖Deploy Guide

We provide two deployment options:
1. Deploy the policy on the [**real robot**](#deploy-on-unitree-robot).  
2. Deploy the policy on your [**workstation**](#deploy-from-your-computer) and control the robot via a wired Ethernet connection.  

## Deploy on Unitree Robot

Run the policy directly **on the robot**.

## Setup
1. Clone our repository and setup the environment on the robot’s onboard computer: see [Basic Setup](../README.md#🛠️Easy-Setup).
2. check [🛠️SDK Installation](#️sdk-installation) for Unitree SDK setup.

Since the G1 has limited computing resources, you need to run `UnitreeCppEnv`. 

Usually, the robot's network interface is `eth0` (for G1). You don't need to modify the config. If you find it doesn't work. see [network configuration](#network-configuration) for help

## Deploy from Your Computer

Run the policy on your computer and control the robot via Ethernet. 

Both `UnitreeEnv` and `UnitreeCppEnv` are supported.

After setup install, connect your robot via Ethernet, then check [network configuration](#network-configuration).

---

## Network Configuration

Refer to [official guide](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/README.md) to connect and find the robot's network interface.

<!-- Run ifconfig to check which network interface is currently connected to the robot.

<div align="center">
<img src="images\net_if.png" alt="network interface" width="70%" >
</div>

Typically, the interface is assigned an IP in the range `192.168.123.XX`. -->

Then edit config to update the `env_type` and `net_if` accordingly:

For example:

**Option 1:** Open [`g1_cfg.py`](robojudo/config/g1/g1_cfg.py) and modify the `g1_real` config.

```python
class g1_real(g1):
    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeEnv",  # For unitree_sdk2py
        # env_type="UnitreeCppEnv", # For unitree_cpp, check README for more details
        unitree=UnitreeEnvCfg.UnitreeCfg(
            net_if="eth0",  # note: change to your network interface
            robot="g1",
            msg_type="hg",
        ),
    )
```
**Option 2:** Go to [g1_real_env_cfg.py](../robojudo/config/g1/env/g1_real_env_cfg.py) and update the global config for g1.

```python
class G1RealEnvCfg(G1EnvCfg, UnitreeEnvCfg):
    env_type: str = UnitreeEnvCfg.model_fields["env_type"].default
    # ====== ENV CONFIGURATION ======
    unitree: UnitreeEnvCfg.UnitreeCfg = UnitreeEnvCfg.UnitreeCfg(
        net_if="eth0", # EDIT HERE
        robot="g1",
        msg_type="hg",
    )
```