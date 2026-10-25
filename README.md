# bilgeval

Evaluation of language models served behind OpenAI-compatible endpoints:
multiple choice by log-probability or by generation, free-response maths,
code execution, instruction following (IFEval, IFBench), function calling
(BFCL), multi-turn judging, long context and typed decisions. Every run
writes a manifest with the exact configuration and fingerprints, per-item
responses and a final report.

## Install

    pip install -e ".[all]"      # ".[all,td]" for typed-decision tasks

## Run

    bilgeval --base-url http://localhost:8000/v1 --model my-model \
        --registry-file my_registry.json my_task

No benchmarks are bundled. Datasets, tasks and suites come from a registry
file (`bilgeval.registry/v1`, defined in `bilgeval/registry_bundle.py`;
`tests/fixtures/registry_v1.json` is a small example). Recipes and
documentation will follow.

## License

Apache License 2.0, see `LICENSE`. The vendored IFEval and IFBench checkers
keep their own notice in `bilgeval/instruction_following/NOTICE.md`.
