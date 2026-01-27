from robojudo.config.config_manager import ConfigManager
from robojudo.config.g1.g1_cfg import g1_goalkeeper_real


def test_goalkeeper_config_loads():
    cfg = ConfigManager(config_name="g1_goalkeeper_real").get_cfg()
    assert cfg.policy.policy_type == "GoalkeeperPolicy"


def test_goalkeeper_real_start_toggle():
    cfg = g1_goalkeeper_real()
    unitree_ctrl = next(ctrl for ctrl in cfg.ctrl if ctrl.ctrl_type == "UnitreeCtrl")
    assert unitree_ctrl.triggers_extra.get("Start") == "[POLICY_TOGGLE]"
