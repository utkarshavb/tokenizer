import os, regex as re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

def refine_boundaries(
    f, file_sz: int, chunk_boundaries: list[int], split_special_token: bytes
) -> list[int]:
    """Makes sure that the chunk boundaries lie on the `split_special_token`"""
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

def pre_tokenize(
    path: str|os.PathLike, regex_pattern: str,
    special_tokens: list[str], split_special_token: bytes
) -> dict[tuple[int, ...], int]:
    file_sz: int = os.path.getsize(path)
    targ_chunk_sz = 64 if file_sz<=4096*1024*1024 else 512
    desired_num_chunks = (file_sz+targ_chunk_sz-1)//targ_chunk_sz

    # initial guesses for chunk boundaries (uniformly placed)
    boundaries = [i*targ_chunk_sz for i in range(desired_num_chunks+1)]
    boundaries[-1] = file_sz
    # refine boundaries
    with open(path, 'rb') as f:
        boundaries = refine_boundaries(f, file_sz, boundaries, split_special_token)
    pat_re = re.compile(regex_pattern)
    
    def pre_tokenize_chunk(interval: tuple[int, int]):
        start, end = interval
        with open(path, 'rb') as f:
            f.seek(start)
            chunk = f.read(end-start).decode("utf-8", errors="ignore")
        split_pat = "|".join(map(re.escape, special_tokens))
        docs = [chunk] if not special_tokens else re.split(split_pat, chunk)
        return  Counter(
            tuple(match.group().encode('utf-8'))
            for doc in docs for match in pat_re.finditer(doc)
        )

    pre_tokens = Counter()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        intervals = zip(boundaries, boundaries[1:])
        for d in executor.map(pre_tokenize_chunk, intervals):
            pre_tokens.update(d)
    return pre_tokens