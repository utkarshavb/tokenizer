# BPE Tokenizer
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This is a pure python, from-scratch implementation for training a Byte-Pair Encoding (BPE) tokenizer. Intended as an educational reference for the optimizations involved in BPE training and inference. This project aims to be readable, hackable, reasonably fast (much faster than any minimal implementations), but still faithful to real-world optimizations.

For questions/doubts, talk to the repository using [DeepWiki](deepwiki.com/utkarshavb/tokenizer) (just replace "github.com" in the repo url with "deepwiki.com") from Devin/Cognition.

## Features
- Fast GPT-4 style regex pre-tokenization with multiprocessing
- Incremental updates to pair frequencies enabling very fast merging
- Decently fast inference code, still aiding in understanding
- Ability to inference with tiktoken

## Additional Notes
I tried representing token sequences as linked-lists instead of tuples which leads to even better big-O complexity but empirical testing revealed that to be slower. Suspicion is that the object overhead in python dominates the theoretical advantage. Even system language implementations (like [rustbpe](github.com/karpathy/rustbpe) and huggingface's tokenizer) seem to use tuple-like data structures instead of linked-lists (probably the caching advantages of an array like data structure dominate pointer chasing).

## Quick Start
### Installation
This project uses [uv](https://docs.astral.sh/uv/) for package management. Install uv via `pip install uv` and then run:
```bash
git clone https://github.com/utkarshavb/tokenizer.git
cd tokenizer
uv sync
```

### Usage
```bash
uv run scripts/train.py path/to/your/corpus --\
    --name="my_tokenizer" \
    --out-dir="trained_tokenizers" \
    --vocab-size=10000 \

uv run scripts/encode.py path/to/your/corpus destination/for/encoded/text.npy --\
    --tokenizer=trained_tokenizers/my_tokenizer.json \
```
This saves the trained tokenizer to the disk at `trained_tokenizers/my_tokenizer.json`, and then uses it to encode the corpus into an array of integers (tokens).

Optionally, we can also run inference with tiktoken:
```bash
uv run scripts/encode.py path/to/your/corpus destination/for/encoded/text --\
    --tokenizer=trained_tokenizers/my_tokenizer.json \
    --use-tiktoken \
```

## Limitations
- Training is slower than Rust/C++ implementations
- Memory usage scales with number of unique pre-tokens
- tiktoken mode requires loading full file into memory
- Full test coverage is pending (as of now)

## Contributing
### Requirements
* Python 3.10+
* uv: `pip install uv`

### Setup
```bash
git clone https://github.com/utkarshavb/tokenizer.git
cd tokenizer
uv sync
```

### Project Structure
```
tokenizer/
├── pyproject.toml
├── scripts/                     # CLI for training and encoding
│   ├── encode.py
│   └── train.py
├── tokenizer/
│   ├── __init__.py
│   ├── core.py                  # main tokenizer interface (inference logic)
│   ├── pre_tokenization.py      # multiprocessing pre-tokenization
│   ├── training.py              # BPE training algorithm
│   └── utils.py                 # utilities for training and save/load
└── uv.lock
```

## Acknowledgements
This repository is a product of following the Stanford [CS336: Language Modelling from Scratch](https://cs336.stanford.edu/) course. The high-level algorithm overview comes from the [assignment 1](https://github.com/stanford-cs336/assignment1-basics/) of this course. A lot of the motivation and inspiration for this repository also comes from [minbpe](github.com/karpathy/minbpe) and [rustbpe](github.com/karpathy/rustbpe). Immense thanks to [Andrej Karpathy](karpathy.ai) for his continued contribution towards open source and education.

LLM assistance note: The code has been written manually and from scratch, but I did use ChatGPT extensively as a reviewer and for consultation on design decisions.

## License
MIT