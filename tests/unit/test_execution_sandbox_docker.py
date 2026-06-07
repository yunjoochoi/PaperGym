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


def test_does_not_forward_provider_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LITELLM_MODEL", "gpt-5")
    monkeypatch.setenv("GYM_LLM_URL", "http://host.docker.internal:9000")
    sb = DockerSandbox(work_root=tmp_path / "run", image="img")
    sb.reset()
    fake = mock.MagicMock(returncode=0, stdout="", stderr="")
    with mock.patch.object(sb_mod.subprocess, "run", return_value=fake) as run:
        sb.run_python("m.py")
    argv = run.call_args.args[0]
    joined = " ".join(argv)
    assert "GYM_LLM_URL" in joined
    assert "OPENAI_API_KEY" not in joined and "sk-test" not in joined
    assert "LITELLM_MODEL" not in joined


def test_docker_adds_host_gateway_and_rewrites_proxy_host(tmp_path, monkeypatch):
    monkeypatch.setenv("GYM_LLM_URL", "http://127.0.0.1:9000")
    sb = DockerSandbox(work_root=tmp_path / "run", image="img")
    sb.reset()
    fake = mock.MagicMock(returncode=0, stdout="", stderr="")
    with mock.patch.object(sb_mod.subprocess, "run", return_value=fake) as run:
        sb.run_python("m.py")
    argv = run.call_args.args[0]
    joined = " ".join(argv)
    assert "--add-host=host.docker.internal:host-gateway" in argv
    assert "host.docker.internal:9000" in joined
    assert "127.0.0.1:9000" not in joined


def test_exec_dockerfile_does_not_install_public_dataset_stack():
    dockerfile = (Path(__file__).resolve().parents[2]
                  / "docker" / "Dockerfile.exec").read_text()
    assert "--no-deps -e ." in dockerfile
    assert "pip install --no-cache-dir datasets" not in dockerfile
    assert "apt-get install" not in dockerfile
