"""설치 단계에서 모든 Numba JIT 커널을 사전 컴파일하여 __pycache__를 생성한다."""
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from localization.lidar_localizer import LidarLocDriver


def main():
    t0 = time.perf_counter()
    print("[warmup] LidarLocDriver + UnifiedCorridorMPC 초기화 및 Numba 사전 컴파일 시작...")
    drv = LidarLocDriver(
        init="start",
        forbid_gt=True,
        inner="drivers.unified_corridor_mpc:UnifiedCorridorMPC",
        profile="profiles/berlin_unified.json",
        map_name="berlin",
    )
    drv.reset()
    dummy_scan = np.full(1080, 5.0, dtype=np.float64)
    drv.act(dummy_scan, *(float("nan"),) * 5, 0.0)
    print(f"[warmup] 완료 ({time.perf_counter() - t0:.2f}s) - 이제 본 실행 시 지연 없이 즉시 시작됩니다.")


if __name__ == "__main__":
    main()
