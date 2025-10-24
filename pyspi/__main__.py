# pyspi/__main__.py
import sys

def main():
    if len(sys.argv) > 1 and sys.argv[1] in {"compute", "visualize"}:
        cmd = sys.argv[1]
        # shift args so the submodule sees its own CLI
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        mod = __import__(f"pyspi.{cmd}", fromlist=["main"])
        mod.main()
    else:
        print("Usage:\n  python -m pyspi compute [--mode dev|paper ...]\n  python -m pyspi visualize [--profile dev|paper ...]")

if __name__ == "__main__":
    main()
