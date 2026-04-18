"""python __main__.py <subcommand> 统一入口"""

import sys


def main():
    commands = {
        "train": "cli.train",
        "finetune": "cli.finetune",
        "generate": "cli.generate",
        "server": "cli.server",
        "train-tokenizer": "scripts.data_prep.train_tokenizer",
        "sample-corpus": "scripts.data_prep.sample_corpus",
        "eval-geo-prompts": "scripts.eval.eval_geo_prompts",
    }

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python __main__.py <command> [args...]")
        print(f"\nCommands: {', '.join(commands)}")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(commands)}")
        sys.exit(1)

    import importlib
    mod = importlib.import_module(commands[cmd])
    sys.argv = sys.argv[1:]
    mod.main()


if __name__ == "__main__":
    main()
