import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prefetch_datasets as pf  # noqa: E402


def test_prefetch_warms_each_task_dataset(monkeypatch, capsys):
    # patch the classmethod so no network/HF access in the test
    monkeypatch.setattr(pf.TASKS["gsm8k_accuracy"], "materialize",
                        classmethod(lambda cls, **k: {"test": 1, "dev": 1}))
    pf.main(["--tasks", "gsm8k_accuracy"])
    assert "materialized gsm8k_accuracy" in capsys.readouterr().out


def test_prefetch_defaults_to_all_registered_tasks(monkeypatch, capsys):
    monkeypatch.setattr(pf.TASKS["gsm8k_accuracy"], "materialize",
                        classmethod(lambda cls, **k: {"test": 1, "dev": 1}))
    pf.main([])  # no --tasks -> all registered
    assert "materialized gsm8k_accuracy" in capsys.readouterr().out
