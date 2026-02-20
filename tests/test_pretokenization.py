import regex as re
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