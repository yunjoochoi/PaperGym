import papergym.agents.accumulator as accumulator_mod


def test_no_tool_baseline_exports_no_tool_schemas():
    assert not hasattr(accumulator_mod, "ACCUMULATOR_TOOLS")


def test_no_tool_baseline_still_exports_accumulator():
    assert hasattr(accumulator_mod, "Accumulator")
