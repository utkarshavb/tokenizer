import os
import regex as re
from tqdm import tqdm
from functools import lru_cache
from collections.abc import Iterable, Iterator
from tokenizer.utils import Pair, load_tokenizer

class Tokenizer:
    def __init__(
        self, merges: list[Pair], pattern: str, special_tokens: list[str]|None = None
    ):
        if special_tokens is None:
            special_tokens = []
        self.vocab = self._build_vocab(merges, special_tokens)
        self.ranks = {pair: 256+i for i, pair in enumerate(merges)}
        self.special_tokens = {
            tok: 256+len(merges)+i for i, tok in enumerate(special_tokens)
        }
        self.split_re = re.compile("(" + "|".join(map(re.escape, special_tokens)) + ")")
        self.PAT_RE = re.compile(pattern)

    def _build_vocab(
        self, merges: list[Pair], special_tokens: list[str]
    ) -> dict[int, bytes]:
        vocab = {i: bytes([i]) for i in range(256)}
        for id1, id2 in merges:
            vocab[len(vocab)] = vocab[id1] + vocab[id2]
        for tok in special_tokens:
            vocab[len(vocab)] = tok.encode('utf-8')
        return vocab

    @classmethod
    def from_files(cls, path: str|os.PathLike, extra_specials: list[str]|None=None):
        merges, pat, special_tokens = load_tokenizer(path)
        if extra_specials is not None:
            special_tokens += extra_specials
        special_tokens = list(set(special_tokens))
        return cls(merges, pat, special_tokens)

    @lru_cache(maxsize=20000)
    def _encode_chunk(self, byte_chunk: bytes) -> list[int]:
        """encodes a single regex chunk using BPE logic"""
        ids = list(byte_chunk)
        while len(ids)>1:
            merge_pair = min(
                zip(ids, ids[1:]), key=lambda pair: self.ranks.get(pair, float("inf"))
            )
            if merge_pair not in self.ranks:
                break
            new_id = self.ranks[merge_pair]
            # replace all occurances of `merge_pair`
            i, new_ids = 0, []
            while i < len(ids):
                if i<len(ids)-1 and (ids[i],ids[i+1])==merge_pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids

    def encode(self, txt: str, allowed_special: bool=True) -> list[int]:
        """
        `allowed_special` allows us to optionally switch-off special-token parsing,
        useful for tokenizing user (attacker) provided text as an example
        """
        ids, docs = [], [txt]
        if allowed_special and self.special_tokens:
            docs = self.split_re.split(txt)
        for doc in docs:
            if doc in self.special_tokens:
                ids.append(self.special_tokens[doc])
            else:
                for match in self.PAT_RE.finditer(doc):
                    byte_chunk = match.group().encode('utf-8')
                    ids.extend(self._encode_chunk(byte_chunk))
        return ids

    def encode_iterable(self, iterable: Iterable[str], allowed_special=True) -> Iterator[int]:
        """memory-efficient tokenization of large files that cannot be loaded into memory"""
        buffer = []
        buffer_size = 0
        for txt in tqdm(iterable, desc="Encoding"):
            buffer.append(txt)
            buffer_size += len(txt)
            if buffer_size >= 1024*1024:
                yield from self.encode("".join(buffer), allowed_special)
                buffer.clear(); buffer_size = 0
        if buffer:
            yield from self.encode("".join(buffer), allowed_special)

    def decode(self, ids: list[int]) -> str:
        byte_seq = b''.join(self.vocab[id] for id in ids)
        txt = byte_seq.decode('utf-8', errors='replace')
        return txt