import argparse

from .code_execution import serve_unix_executor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bilgeval credential-free HumanEval execution service."
    )
    parser.add_argument("--socket", required=True)
    args = parser.parse_args()
    serve_unix_executor(args.socket)


if __name__ == "__main__":
    main()
