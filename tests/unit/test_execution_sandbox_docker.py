from pathlib import Path
from unittest import mock
from papergym.execution import sandbox as sb_mod
from papergym.execution.sandbox import DockerSandbox


def test_run_python_invokes_docker_run_with_mount(tmp_path):
    sb = DockerSandbox(work_root=tmp_path / "run", image="papergym-exec:test")
    sb.reset()
    fake = mock.MagicMock(returncode=0, stdout="ok", stderr="")
    with mock.patch.object(sb_mod.subprocess, "run", return_value=fake) as run:
        rc, out, err = sb.run_python("m.py", timeout=120)
    assert rc == 0 and out == "ok"
    argv = run.call_args.args[0]
    assert argv[0] == "docker" and "run" in argv
    assert any(str(sb.work_root) + ":/work" in a for a in argv)
    assert "papergym-exec:test" in argv


def test_forwards_provider_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LITELLM_MODEL", "gpt-5")
    sb = DockerSandbox(work_root=tmp_path / "run", image="img")
    sb.reset()
    fake = mock.MagicMock(returncode=0, stdout="", stderr="")
    with mock.patch.object(sb_mod.subprocess, "run", return_value=fake) as run:
        sb.run_python("m.py")
    argv = run.call_args.args[0]
    joined = " ".join(argv)
    assert "OPENAI_API_KEY" in joined and "LITELLM_MODEL" in joined
