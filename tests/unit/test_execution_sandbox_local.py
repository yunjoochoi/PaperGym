from papergym.execution.sandbox import LocalSandbox


def test_init_is_cheap_no_io(tmp_path):
    sb = LocalSandbox(work_root=tmp_path / "run1")
    assert not (tmp_path / "run1").exists()           # __init__ does no I/O


def test_write_run_read_roundtrip(tmp_path):
    with LocalSandbox(work_root=tmp_path / "run1") as sb:
        sb.write_file("m.py", "open('out.txt','w').write('hi'); print('done')")
        rc, out, err = sb.run_python("m.py", timeout=30)
        assert rc == 0 and "done" in out
        assert sb.read_file("out.txt") == "hi"


def test_timeout_returns_124(tmp_path):
    with LocalSandbox(work_root=tmp_path / "run2") as sb:
        sb.write_file("loop.py", "while True: pass")
        rc, out, err = sb.run_python("loop.py", timeout=1)
        assert rc == 124
