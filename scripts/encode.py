import argparse
import numpy as np
from tokenizer.core import Tokenizer
from tokenizer.utils import load_tiktoken_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="path to a text file to encode")
    parser.add_argument("out", type=str, help="where to save the encoded text (.npy format)")
    parser.add_argument("--tokenizer", type=str, required=True, help="path to a tokenizer.json file")
    parser.add_argument("--use-tiktoken", action="store_true", help="encodes with tiktoken. NOTE: requires loading the full file in memory")
    parser.add_argument("--special-tokens", type=str, nargs="*", default=[], help="tokens to treat as special while tokenizing")
    args = parser.parse_args()

    if args.use_tiktoken:
        tokenizer = load_tiktoken_tokenizer(args.tokenizer, extra_specials=args.special_tokens)
        with open(args.text, "r", encoding="utf-8") as f:
            text = f.read()
        encoded = tokenizer.encode(text, allowed_special="all")
        vocab_size = tokenizer.n_vocab
    else:
        tokenizer = Tokenizer.from_files(args.tokenizer, extra_specials=args.special_tokens)
        with open(args.text, "r", encoding="utf-8") as f:
            encoded = list(tokenizer.encode_iterable(f))
        vocab_size = len(tokenizer.vocab)

    dtype = np.uint16 if vocab_size<=65_535 else np.uint32
    encoded_arr = np.array(encoded, dtype=dtype)
    np.save(args.out, encoded_arr)