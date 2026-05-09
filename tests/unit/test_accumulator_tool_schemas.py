from papergym.agents.accumulator import ACCUMULATOR_TOOLS


def test_has_required_tools():
    names = [t["function"]["name"] for t in ACCUMULATOR_TOOLS]
    assert set(names) == {"Read", "Grep", "Bash"}


def test_no_done_tool():
    """`done` was removed; the agent terminates by emitting JSON content
    instead of invoking a structured-output tool."""
    names = [t["function"]["name"] for t in ACCUMULATOR_TOOLS]
    assert "done" not in names
