from robojudo.controller.ctrl_cfgs import GoalkeeperBallDdsCtrlCfg


class G1GoalkeeperBallDdsCtrlCfg(GoalkeeperBallDdsCtrlCfg):
    domain_id: int = 0
    topic_name: str = "detectionresults"
    poll_interval: float = 0.01
    timeout_s: float = 0.3
