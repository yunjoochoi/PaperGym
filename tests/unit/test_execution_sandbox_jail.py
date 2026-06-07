import pytest
from papergym.execution.sandbox import LocalSandbox, DockerSandbox, SandboxPathError


def test_local_rejects_parent_traversal(tmp_path):
    with LocalSandbox(work_root=tmp_path / "run") as sb:
        with pytest.raises(SandboxPathError):
            sb.write_file("../escape.txt", "x")
        with pytest.raises(SandboxPathError):
            sb.read_file("../../etc/passwd")
        with pytest.raises(SandboxPathError):
            sb.run_python("../m.py")


def test_local_rejects_absolute_path(tmp_path):
    with LocalSandbox(work_root=tmp_path / "run") as sb:
        with pytest.raises(SandboxPathError):
            sb.write_file("/tmp/x.txt", "x")


def test_local_allows_in_workspace(tmp_path):
    with LocalSandbox(work_root=tmp_path / "run") as sb:
        sb.write_file("sub/m.py", "print('ok')")
        rc, out, _ = sb.run_python("sub/m.py")
        assert rc == 0 and "ok" in out


def test_docker_forwards_proxy_url_not_provider_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("GYM_LLM_URL", "http://host.docker.internal:9000")
    sb = DockerSandbox(work_root=tmp_path / "run", image="img")
    flags = sb._env_flags()
    joined = " ".join(flags)
    assert "GYM_LLM_URL" in joined
    assert "OPENAI_API_KEY" not in joined and "sk-secret" not in joined
