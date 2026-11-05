"""Download PaperBananaBench. Equivalent to: paperbanana prepare-bench [--tasks diagram plot]."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperbanana.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["prepare-bench", *sys.argv[1:]]))
