import sys
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_main_writes_method_json_to_stdout(tmp_path: Path, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import run_synthesis as rs

    paraphrases = {d: "x" for d in
                   ["LLM_NLP", "MULTIMODAL", "CV", "RL", "IR_REC", "SPEECH", "ROBOTICS"]}
    fake_para = MagicMock()
    fake_para.return_value.run.return_value = {"essence": "e",
                                                "paraphrases": paraphrases}
    fake_synth = MagicMock()
    fake_synth.return_value.run.return_value = {"method": "M",
                                                 "rationale": "R",
                                                 "inspired_by": []}

    with patch.object(rs, "Paraphraser", fake_para), \
         patch.object(rs, "Synthesizer", fake_synth), \
         patch.object(rs.LibraryStore, "retrieve_cross_domain", return_value=[]), \
         patch("os.environ", {"OPENAI_API_KEY": "sk-test", "LITELLM_MODEL": "gpt-5"}):
        rs.main(argv=["--query", "long-context inference",
                      "--library-root", str(tmp_path)])

    out = capsys.readouterr().out
    assert '"method": "M"' in out
