from papergym.domain import Domain
from papergym.library.seed import Seed, new_seed_id


def test_seed_round_trip_to_dict():
    s = Seed(seed_id="abc123",
             problem="long-context attention is quadratic",
             method="sparse attention with sliding window",
             domain=Domain.LLM_NLP,
             paper_title="Longformer",
             paper_id="2004.05150")
    d = s.to_dict()
    assert d["domain"] == "LLM_NLP"
    s2 = Seed.from_dict(d)
    assert s2 == s


def test_new_seed_id_is_12_hex_chars():
    sid = new_seed_id()
    assert len(sid) == 12
    int(sid, 16)  # parses as hex
