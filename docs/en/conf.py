import sys
from pathlib import Path

repository_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository_root))

from docs.shared_conf import *  # noqa: F401,F403

language = "en"
