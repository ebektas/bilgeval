"""Deterministic suite sampling. Default preserves bilgeval's existing selection.

stable_hash takes example objects and a run with sample_seed, dataset_name,
max_examples. sample_examples also accepts JSON dictionaries and dispatches on
the optional run.sample_strategy. Stratum remainder order is SHA256(name UTF-8),
without seed; within-stratum and final order use bilgeval's seeded hash.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from types import SimpleNamespace


def example_sample_key(example, index: int) -> str:
    example_id = getattr(example, "id", "")
    if example_id:
        return str(example_id)
    raw = getattr(example, "raw", None)
    if raw is not None:
        try:
            return json.dumps(raw, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(raw)
    return str(index)


def stable_hash(examples: list, run: ResolvedBenchmarkRun) -> list:
    sample_seed = run.sample_seed
    if not sample_seed:
        return examples[: run.max_examples] if run.max_examples is not None else examples
    ranked = sorted(
        enumerate(examples),
        key=lambda item: hashlib.sha256(
            f"{sample_seed}\0{run.dataset_name}\0{example_sample_key(item[1], item[0])}".encode("utf-8")
        ).hexdigest(),
    )
    sampled = [example for _, example in ranked]
    return sampled[: run.max_examples] if run.max_examples is not None else sampled


def stratified_stable_hash(examples: list, run) -> list:
    """Equal seats recursively per label, then subtype; redistribute among siblings.

    Requires a nonempty seed, unique IDs and nonempty string sample_stratum.
    Output is globally ordered by the same seeded hash as stable_hash.
    """
    if not run.sample_seed:
        raise ValueError('stratified_stable_hash requires sample_seed')
    if run.max_examples is not None and (type(run.max_examples) is not int or run.max_examples < 0):
        raise ValueError('max_examples must be a nonnegative integer or None')
    groups = defaultdict(list)
    ids = set()
    for example in examples:
        example_id = getattr(example, 'id', None)
        metadata = getattr(example, 'metadata', {})
        stratum = metadata.get('sample_stratum') if isinstance(metadata, dict) else None
        if not isinstance(stratum, str) or not stratum.strip():
            raise ValueError('metadata.sample_stratum must be a nonempty string')
        if not isinstance(example_id, str) or not example_id or example_id in ids:
            raise ValueError('stratified sampling requires unique nonempty string IDs')
        ids.add(example_id)
        groups[stratum].append(example)
    budget = len(examples) if run.max_examples is None else min(run.max_examples, len(examples))
    def select_level(grouped, seats, depth):
        children = defaultdict(dict)
        for path, members in grouped.items():
            children[path[depth]][path] = members
        counts = {name: sum(len(v) for v in paths.values()) for name, paths in children.items()}
        order = sorted(children, key=lambda name: (hashlib.sha256(name.encode('utf-8')).hexdigest(), name))
        quotas = dict.fromkeys(order, 0)
        remaining = seats
        while remaining:
            active = [name for name in order if quotas[name] < counts[name]]
            base, extra = divmod(remaining, len(active))
            for i, name in enumerate(active):
                take = min(base if base else int(i < extra), counts[name] - quotas[name])
                quotas[name] += take
                remaining -= take
        result = []
        for name in order:
            paths = children[name]
            terminal = [path for path in paths if len(path) == depth + 1]
            if terminal:
                if len(paths) != 1:
                    raise ValueError('sample_stratum cannot be both a leaf and parent')
                result.extend(stable_hash(paths[terminal[0]], SimpleNamespace(
                    sample_seed=run.sample_seed, dataset_name=run.dataset_name,
                    max_examples=quotas[name])))
            else:
                result.extend(select_level(paths, quotas[name], depth + 1))
        return result

    paths = {tuple(name.split('|')): members for name, members in groups.items()}
    if any(any(not part.strip() for part in path) for path in paths):
        raise ValueError('sample_stratum hierarchy contains an empty level')
    selected = select_level(paths, budget, 0) if paths else []
    return stable_hash(selected, SimpleNamespace(sample_seed=run.sample_seed,
                       dataset_name=run.dataset_name, max_examples=None))


def sample_examples(examples: list, run) -> list:
    """Select using run.sample_strategy, default stable_hash; preserve row objects."""
    if run.max_examples is not None and (type(run.max_examples) is not int or run.max_examples < 0):
        raise ValueError('max_examples must be a nonnegative integer or None')
    strategy = getattr(run, 'sample_strategy', 'stable_hash')
    methods = {'stable_hash': stable_hash, 'stratified_stable_hash': stratified_stable_hash}
    if strategy not in methods:
        raise ValueError(f'Unknown sample_strategy: {strategy}')
    adapted = [SimpleNamespace(id=row.get('id',''), raw=row, metadata=row.get('metadata',{}))
               if isinstance(row, dict) else row for row in examples]
    originals = {id(example): row for example, row in zip(adapted, examples)}
    return [originals[id(example)] for example in methods[strategy](adapted, run)]
