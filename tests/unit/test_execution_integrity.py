from papergym.execution.integrity import scan_for_leakage


def test_flags_dataset_reload_and_direct_provider():
    code = ("from datasets import load_dataset\n"
            "import openai\n"
            "ds = load_dataset('openai/gsm8k','main',split='test')\n")
    flags = scan_for_leakage(code)
    assert any("load_dataset" in f for f in flags)
    assert any("openai" in f for f in flags)


def test_clean_code_has_no_flags():
    code = ("from papergym.execution.gym_client import metered_llm_call\n"
            "metered_llm_call([{'role':'user','content':'hi'}])\n")
    assert scan_for_leakage(code) == []


def test_flags_dynamic_import_network_and_shell_escape():
    code = ("mod = __import__('datasets')\n"
            "import socket\n"
            "import subprocess\n"
            "subprocess.run(['python','-m','pip','install','datasets'])\n")
    flags = scan_for_leakage(code)
    assert any("dynamic-import" in f for f in flags)
    assert any("socket-network" in f for f in flags)
    assert any("subprocess-shell" in f for f in flags)
    assert any("package-install" in f for f in flags)
