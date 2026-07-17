from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CV_SRC = ROOT.parent / "cv_processor" / "src"
for path in (SRC, CV_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
