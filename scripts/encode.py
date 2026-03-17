from pathlib import Path
import argparse
import numpy as np
from tokenizer.core import Tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="path to a text file to encode")
    parser.add_argument("out", type=str, help="where to save the encoded text (.npy format)")
    parser.add_argument("--tokenizer", type=str, required=True, help="path to a tokenizer.json file")
    parser.add_argument("--special-tokens", type=str, nargs="*", default=[], help="tokens to treat as special while tokenizing")
    parser.add_argument("--dtype", type=str, default="uint16", help="data type of the output array")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.tokenizer, extra_specials=args.special_tokens)
    with open(args.text, 'r', encoding="utf-8") as f:
        encoded = list(tokenizer.encode_iterable(f))

    encoded_arr = np.array(encoded, dtype=args.dtype)
    np.save(args.out, encoded_arr)