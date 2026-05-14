import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from papergym.agents import Paraphraser, PromptLoader, Synthesizer
from papergym.library import LibraryStore
from papergym.llm import LLMClient

load_dotenv(override=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--library-root", required=True, type=Path)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--natural-domain", default=None)
    args = p.parse_args(argv)

    llm = LLMClient()
    paraphrase_prompts = PromptLoader("papergym.agents.paraphraser")
    synth_prompts = PromptLoader("papergym.agents.synthesizer")
    library = LibraryStore.open_merged(args.library_root)

    events_dir = args.library_root / "synthesis" / time.strftime("%Y%m%d-%H%M%S")
    para = Paraphraser(llm, paraphrase_prompts).run(args.query)
    triples = library.retrieve_cross_domain(
        para["paraphrases"],
        raw_query=args.query,
        natural_domain=args.natural_domain,
        llm=llm,
        k=args.k,
    )
    seeds = [s for s, _, _ in triples]
    lenses = [lens for _, _, lens in triples]
    result = Synthesizer(llm, synth_prompts).run(
        args.query, seeds=seeds, lenses=lenses, events_dir=events_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
