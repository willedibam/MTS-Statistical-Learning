# src/spimts/__main__.py
import sys

def main():
    if len(sys.argv) > 1 and sys.argv[1] in {"compute","visualize"}:
        cmd = sys.argv[1]
        sys.argv = [sys.argv[0]] + sys.argv[2:]   # shift args for subcommand
        mod = __import__(f"spimts.{cmd}", fromlist=["main"])
        mod.main()
    else:
        print("Usage:\n  python -m spimts compute [--mode ...]\n  python -m spimts visualize [--profile ...]")

if __name__ == "__main__":
    main()
