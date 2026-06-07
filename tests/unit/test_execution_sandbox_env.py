import json
from papergym.execution.sandbox import LocalSandbox


def test_local_subprocess_env_drops_provider_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-secret")
    monkeypatch.setenv("GYM_LLM_URL", "http://localhost:1234")
    with LocalSandbox(work_root=tmp_path / "run") as sb:
        sb.write_file("m.py",
                      "import os,json;"
                      "json.dump({k:os.environ.get(k) for k in "
                      "['OPENAI_API_KEY','ANTHROPIC_API_KEY','GYM_LLM_URL']},"
                      "open('out.json','w'))")
        sb.run_python("m.py")
        seen = json.loads(sb.read_file("out.json"))
    assert seen["OPENAI_API_KEY"] is None
    assert seen["ANTHROPIC_API_KEY"] is None
    assert seen["GYM_LLM_URL"] == "http://localhost:1234"
