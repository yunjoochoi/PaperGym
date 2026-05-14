from papergym.domain import Domain, DOMAIN_FIELDS, DOMAIN_OVERRIDES


def test_seven_domains_no_generative():
    names = {d.value for d in Domain}
    assert names == {"LLM_NLP", "MULTIMODAL", "CV", "RL",
                     "IR_REC", "SPEECH", "ROBOTICS"}


def test_domain_fields_cover_all_domains():
    for d in Domain:
        assert d in DOMAIN_FIELDS
        assert isinstance(DOMAIN_FIELDS[d], list) and DOMAIN_FIELDS[d]


def test_overrides_route_vlm_to_multimodal():
    assert DOMAIN_OVERRIDES["Vision and Language"] == Domain.MULTIMODAL
    assert DOMAIN_OVERRIDES["Visual Question Answering"] == Domain.MULTIMODAL
    assert DOMAIN_OVERRIDES["Image Captioning"] == Domain.MULTIMODAL
