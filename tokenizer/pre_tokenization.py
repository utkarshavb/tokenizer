import os, regex as re
from collections import Counter
from typing import IO
from concurrent.futures import ProcessPoolExecutor
import atexit

def refine_boundaries(
    f, file_sz: int, chunk_boundaries: list[int], split_special_token: bytes
) -> list[int]:
    """
    Aligns chunk boundaries to occur at the beginning of `split_special_token`.
    This ensures that each chunk is a valid UTF-8 sequence (assuming the
    initial corpus is valid UTF-8).
    NOTE: This assumes that `split_special_token` is frequent enough in the
    corpus to not result in monster chunk sizes.
    """
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    for bi in range(1, len(chunk_boundaries)-1):
        initial_position = chunk_boundaries[bi]
        f.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = f.read(mini_chunk_size)  # Read a mini chunk
            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_sz
                break
            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    # Make sure all boundaries are unique, but might be fewer than num_chunks
    return sorted(set(chunk_boundaries))

# worker globals (set once per child process) 
_FILE: IO[bytes]|None = None
_PAT_RE: re.Pattern|None = None
_SPLIT_RE: re.Pattern|None = None

def _init_worker(path: str|os.PathLike, regex_pattern: str, special_tokens: list[str]):
    global _FILE, _PAT_RE, _SPLIT_RE

    _FILE = open(path, "rb")
    atexit.register(_FILE.close)
    _PAT_RE = re.compile(regex_pattern)
    if special_tokens:
        split_pat = "|".join(map(re.escape, special_tokens))
        _SPLIT_RE = re.compile(split_pat)
    else:
        _SPLIT_RE = None

def _pre_tokenize_chunk(interval: tuple[int, int]):
    assert _FILE is not None and _PAT_RE is not None, "Worker not initialzed"
    start, end = interval
    _FILE.seek(start)
    chunk = _FILE.read(end-start).decode("utf-8", errors="ignore")
    if _SPLIT_RE:
        docs = _SPLIT_RE.split(chunk)
    else:
        docs = [chunk]
    return  Counter(
        m.group().encode('utf-8') for doc in docs for m in _PAT_RE.finditer(doc)
    )

def pre_tokenize(
    path: str|os.PathLike, regex_pattern: str,
    special_tokens: list[str], split_special_token: bytes
) -> Counter[bytes]:
    # chunking heuristics for multiprocessing
    file_sz: int = os.path.getsize(path)
    num_workers = os.cpu_count() or 4
    num_chunks = 8*num_workers
    targ_chunk_sz = (file_sz+num_chunks-1)//num_chunks

    # refine boundaries
    boundaries = [i*targ_chunk_sz for i in range(num_chunks+1)]
    boundaries[-1] = file_sz
    with open(path, 'rb') as f:
        boundaries = refine_boundaries(f, file_sz, boundaries, split_special_token)

    pre_tokens = Counter()
    with ProcessPoolExecutor(
        max_workers=num_workers, initializer=_init_worker,
        initargs=(path, regex_pattern, special_tokens)
    ) as ex:
        intervals = zip(boundaries, boundaries[1:])
        for partial in ex.map(_pre_tokenize_chunk, intervals, chunksize=4):
            pre_tokens.update(partial)
    return pre_tokens