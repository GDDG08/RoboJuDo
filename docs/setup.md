<h1 align="center"><b>Setup for Unitree Robots  </b></h1>

RoboJuDo has two environments for unitree robots:

- **`UnitreeEnv`**: Based on `unitree_sdk2py`.
    - Support both  `UnitreeH1` and `UnitreeG1`.
    - May endure performance issues on `UnitreeG1` due to limited computing power.
- **`UnitreeCppEnv`**: Based on [UnitreeCpp](https://github.com/GDDG08/unitree_cpp)
    - Support `UnitreeG1`.
    - Can be deployed onboard `UnitreeG1` pc2. It is much faster.

We provide two recommended deployment options:
1. Deploy the policy on the [**real robot**](#deployment-on-unitree-robot).  
2. Deploy the policy on your [**workstation**](#deployment-from-your-computer) and control the robot via a wired Ethernet connection.  

# Deploy on Unitree Robot

Run the policy directly on the **real robot**.

## Environment Setup
First, make sure you have clone our repository and set up the basic environment on the robot’s onboard computer: see [Basic Setup](../README.md#1️⃣-basic-setup).

### For Unitree G1
Since the G1 has limited computing resources, you need to run `UnitreeCppEnv`. 

1. Install the official SDK: [UnitreeCppSDK](https://github.com/unitreerobotics/unitree_sdk2)
2. and then install our `unitree_cpp` package:
    ```bash
    python submodule_install.py unitree_cpp
    ```
### For Unitree H1

Install [`unitree_sdk2py`](https://github.com/unitreerobotics/unitree_sdk2_python)

## Network Configuration

Usually, the robot's network interface is `eth0`. You don't need to modify the params. If you find it doesn't work. see [network configuration](#network-configuration-1) for help

# Deploy from Your Computer

Run the policy on your computer and control the robot via Ethernet. 

Both the `unitree_cpp` and `unitree_sdk2py` are supported. If you want to use `unitree_sdk2py`, you need to install the official SDK: [`unitree_sdk2py`](https://github.com/unitreerobotics/unitree_sdk2_python)


## Network Configuration

Refer to [here](https://github.com/unitreerobotics/unitree_rl_gym/tree/main/deploy/deploy_real) to connect and find the robot's network interface.

<!-- Run ifconfig to check which network interface is currently connected to the robot.

<div align="center">
<img src="images\net_if.png" alt="network interface" width="70%" >
</div>

Typically, the interface is assigned an IP in the range `192.168.123.XX`. -->

Then open [`robojudo\config\g1\env\g1_real_env_cfg.py`](../robojudo/config/g1/env/g1_real_env_cfg.py) and update the `net_if` interface parameters accordingly:

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