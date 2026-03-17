import os
import time
from contextlib import contextmanager
from tqdm import tqdm
from heapq import heapify, heappop, heappush
from tokenizer.pre_tokenization import pre_tokenize
from tokenizer.utils import (
    Pair, Stats, TokenSeq, init_pair_stats, PAT, ENDOFTEXT
)

@contextmanager
def log_time(verb: str, noun: str):
    print(verb)
    t0 = time.time()
    yield
    dt = time.time()-t0
    print(f"{noun} done! Time taken: {dt:.3f} seconds\n")

def _pop_best_pair(heap: list, pair_stats: dict[Pair, Stats]) -> Pair|None:
    # lazy pop as heap entries can become stale after count updates
    while heap:
        neg_c, neg_id1, neg_id2 = heappop(heap)
        merge_pair = (-neg_id1, -neg_id2)
        stats = pair_stats[merge_pair]
        if stats.count>0 and stats.count==-neg_c:
            return merge_pair
    return None

def train(
    path: str|os.PathLike, vocab_size: int, regex_pattern: str=PAT,
    split_special_token: str=ENDOFTEXT, special_tokens: list[str]|None=None
) -> tuple[list[Pair], list[str]]:
    """
    Args:
        path: path to the training corpus
        vocab_size: vocabulary size of the trained tokenizer
        out: path to save the trained tokenizer to
        regex_pattern: pattern for the initial coarse grained tokenization (pre-tokenization)
        of the corpus (default is the GPT-4 pattern)
        split_special_token: token that delimits documents in the corpus (default: `<|endoftext|>`)
        special_tokens: a list of tokens to preserve from splitting and merging. They are removed
        from the corpus before training. The `split_special_token` is included by default
    """
    special_tokens = [] if special_tokens is None else list(special_tokens)
    if split_special_token not in special_tokens:
        special_tokens.append(split_special_token)
    
    assert vocab_size >= 256+len(special_tokens)

    with log_time("Pre-tokenizing...", "Pre-tokenization"):
        pre_tokens = pre_tokenize(
            path, regex_pattern, special_tokens=special_tokens,
            split_special_token=split_special_token.encode()
        )

    tokseqs = [TokenSeq(tuple(byte_seq), count) for byte_seq, count in pre_tokens.items()]
    with log_time("Initializing pair statistics...", "Stat initialization"):
        pair_stats = init_pair_stats(tokseqs)
    heap = [(-stats.count, -id1, -id2) for (id1, id2), stats in pair_stats.items()]
    heapify(heap)

    # Go!
    merges: list[Pair] = []
    merge_ids = range(256, vocab_size-len(special_tokens))

    with log_time("Training...", "Training"):
        for merge_id in tqdm(merge_ids, desc="Merging"):
            merge_pair = _pop_best_pair(heap, pair_stats)
            if merge_pair is None:
                break
            stats = pair_stats[merge_pair]
            
            affected_pairs: set[Pair] = set()
            for idx in stats.locs:
                tokseq = tokseqs[idx]
                deltas = tokseq.merge(merge_pair, merge_id)

                # update global stats of the affected pairs
                for affected_pair, delta in deltas:
                    tot_delta = delta*tokseq.count
                    affected_stats = pair_stats[affected_pair]
                    affected_stats.count += tot_delta
                    if delta == +1:
                        affected_stats.locs.add(idx)
                    affected_pairs.add(affected_pair)

            # update heap
            for id1, id2 in affected_pairs:
                cnt = pair_stats[(id1, id2)].count
                if cnt < 0:
                    continue
                heappush(heap, (-cnt, -id1, -id2))

            merges.append(merge_pair)

    return merges, special_tokens