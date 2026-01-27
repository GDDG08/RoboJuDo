from robojudo.pipeline.rl_pipeline import _toggle_policy_enabled


def test_toggle_policy_enabled():
    assert _toggle_policy_enabled(False, []) is False
    assert _toggle_policy_enabled(False, ["[POLICY_TOGGLE]"]) is True
    assert _toggle_policy_enabled(True, ["[POLICY_TOGGLE]"]) is False
