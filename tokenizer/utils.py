import os
import json
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import tiktoken

PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
ENDOFTEXT = "<|endoftext|>"

type Pair = tuple[int, int]

@dataclass(slots=True)
class TokenSeq:
    seq: tuple[int, ...]
    count: int

    def merge(self, pair: Pair, new_id: int) -> list[tuple[Pair, int]]:
        new_seq = []
        deltas: list[tuple[Pair, int]] = []
        i, n = 0, len(self.seq)
        while i < n:
            if i<n-1 and (self.seq[i], self.seq[i+1])==pair:
                left = self.seq[i-1] if i>0 else None
                right = self.seq[i+2] if i<n-2 else None
                if left is not None:
                    deltas.append(((left, pair[0]), -1))
                    deltas.append(((left, new_id), +1))
                deltas.append((pair, -1))
                if right is not None:
                    deltas.append(((pair[1], right), -1))
                    deltas.append(((new_id, right), +1))
                new_seq.append(new_id)
                i += 2
            else:
                new_seq.append(self.seq[i])
                i += 1
        self.seq = tuple(new_seq)
        return deltas

@dataclass(slots=True)
class Stats:
    count: int = 0
    # locs can potentially contain stale location for pairs
    locs: set[int] = field(default_factory=set)

def init_pair_stats(tokseqs: list[TokenSeq]) -> defaultdict[Pair, Stats]:
    stats = defaultdict(Stats)
    for idx in range(len(tokseqs)):
        seq = tokseqs[idx].seq
        count = tokseqs[idx].count
        for i in range(1, len(seq)):
            pair: Pair = (seq[i-1], seq[i])
            stats[pair].count += count
            stats[pair].locs.add(idx)
    return stats

def save_tokenizer(
    merges:list[Pair], pat: str, special_tokens: list[str], out: str|os.PathLike
):
    data = dict(
        pattern = pat, special_tokens=special_tokens, merges=merges
    )
    with open(out, "w") as f:
        json.dump(data, f, indent=4)

def load_tokenizer(path: str|os.PathLike) -> tuple[list[Pair], str, list[str]]:
    with open(path, "r") as f:
        data = json.load(f)
    merges = [tuple(pair) for pair in data["merges"]]
    pat = data["pattern"]
    special_tokens = data["special_tokens"]
    return merges, pat, special_tokens

def load_tiktoken_tokenizer(
    path: str|os.PathLike, extra_specials: list[str]|None=None
) -> tiktoken.Encoding:
    if not isinstance(path, Path):
        path = Path(path)
    merges, pat, special_tokens = load_tokenizer(path)
    if extra_specials is not None:
        special_tokens += extra_specials
    special_tokens = list(set(special_tokens))
    special_token_dict = {
        tok: 256+len(merges)+i for i, tok in enumerate(special_tokens)
    }
    vocab = {i: bytes([i]) for i in range(256)}
    for i, (id1, id2) in enumerate(merges):
        vocab[256+i] = vocab[id1]+vocab[id2]
    mergeable_ranks = {
        vocab[256+i]: 256+i for i in range(len(merges))
    }
    tokenizer = tiktoken.Encoding(
        name=path.stem, mergeable_ranks=mergeable_ranks,
        pat_str=pat, special_tokens=special_token_dict,
    )
    return tokenizer