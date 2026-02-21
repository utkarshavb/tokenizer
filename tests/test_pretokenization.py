import regex as re, io
from collections import Counter
from tokenizer.pre_tokenization import refine_boundaries, pre_tokenize

PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
PAT_RE = re.compile(PAT)
EOT = "<|endoftext|>"

def reference_pre_tokenize(path: str, special_tokens: list[str]) -> Counter[bytes]:
    with open(path, "r") as f:
        text = f.read()
    
    split_pat = "|".join(map(re.escape, special_tokens))
    docs = [text] if not special_tokens else re.split(split_pat, text)
    return Counter(m.group().encode() for doc in docs for m in PAT_RE.finditer(doc))

def test_serial_parallel_implementation_equivalence():
    path = "data/owt_valid.txt"
    special_tokens = [EOT]
    
    reference_pre_tokens = reference_pre_tokenize(path, special_tokens)
    pre_tokens = pre_tokenize(path, PAT, special_tokens, EOT.encode())

    assert reference_pre_tokens == pre_tokens

def test_empty_file():
    path = "tests/files/empty.txt"
    assert pre_tokenize(path, PAT, [], EOT.encode()) == Counter()

def test_chunk_boundaries():
    split_special_token = EOT.encode()
    tlen = len(split_special_token)

    data = (
        b"h"*100 + split_special_token + b"!"*1000 +
        split_special_token + split_special_token +
        b"end"*100 + b"of"*1000
    )
    f = io.BytesIO(data)
    file_sz = len(data)
    bad_boundaries = [0] + [99, 105, 1100+2*tlen+3] + [file_sz]
    boundaries = refine_boundaries(f, file_sz, bad_boundaries, split_special_token)

    assert boundaries[0] == 0
    assert boundaries[-1] == file_sz
    assert boundaries == sorted(boundaries)
    assert len(boundaries) == len(set(boundaries))

    for i in range(1, len(boundaries)-1):
        b = boundaries[i]
        assert data[b:b+tlen] == split_special_token