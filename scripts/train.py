import argparse
from pathlib import Path
from tokenizer.utils import PAT, ENDOFTEXT, save_tokenizer
from tokenizer.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=str)
    parser.add_argument("--name", type=str, required=True, help="name to give the tokenizer")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("-v", "--vocab-size", type=int, default=257)
    parser.add_argument("--regex-pattern", type=str, default=PAT, help="pattern for the initial coarse grained tokenization (pre-tokenization) of the corpus (default is the GPT-4 pattern)")
    parser.add_argument("--split-special-token", type=str, default=ENDOFTEXT, help="token that delimits documents in the corpus (default: `<|endoftext|>`)")
    parser.add_argument("--special-tokens", nargs="*", type=str, default=[])
    args = parser.parse_args()

    merges, special_tokens = train(
        args.corpus, vocab_size=args.vocab_size, regex_pattern=args.regex_pattern,
        split_special_token=args.split_special_token, special_tokens=args.special_tokens
    )

    out = Path(args.out_dir)/f"{args.name}.json"
    save_tokenizer(merges, args.regex_pattern, special_tokens=special_tokens, out=out)