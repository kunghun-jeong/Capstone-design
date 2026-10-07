import sys
from pathlib import Path

# 패키지 설치(pyproject)가 아직 없어서 src/를 직접 import 경로에 올린다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
