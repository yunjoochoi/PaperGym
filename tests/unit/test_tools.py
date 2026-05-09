from pathlib import Path

from papergym.tools import bash, grep, read


def test_read_returns_line_numbered_content(tmp_path: Path):
    target = tmp_path / "paper.md"
    target.write_text("alpha\nbeta\ngamma\n")
    out = read(str(target))
    assert "alpha" in out
    assert "beta" in out
    assert out.splitlines()[0].endswith("\talpha")


def test_read_respects_offset_and_limit(tmp_path: Path):
    target = tmp_path / "paper.md"
    target.write_text("\n".join(f"line{i}" for i in range(10)))
    out = read(str(target), offset=5, limit=2)
    lines = out.splitlines()
    assert len(lines) == 2
    assert "line5" in lines[0]
    assert "line6" in lines[1]


def test_bash_runs_in_paper_dir(tmp_path: Path):
    (tmp_path / "marker.txt").write_text("hi")
    out = bash("ls", paper_dir=tmp_path)
    assert "exit=0" in out
    assert "marker.txt" in out


def test_bash_propagates_nonzero_exit(tmp_path: Path):
    out = bash("exit 7", paper_dir=tmp_path)
    assert "exit=7" in out


def test_grep_returns_matches(tmp_path: Path):
    (tmp_path / "a.txt").write_text("foo\nbar\nfoobar\n")
    out = grep("foo", ".", paper_dir=tmp_path)
    assert "foo" in out


def test_grep_returns_no_matches_marker(tmp_path: Path):
    (tmp_path / "a.txt").write_text("nothing here")
    out = grep("absent_pattern_xyz", ".", paper_dir=tmp_path)
    assert out == "(no matches)"


def test_each_tool_carries_an_openai_schema():
    for fn in (read, grep, bash):
        spec = fn.schema
        assert spec["type"] == "function"
        assert "name" in spec["function"]
        assert "parameters" in spec["function"]
