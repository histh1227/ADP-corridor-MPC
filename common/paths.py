"""경로 자동 탐색 유틸리티.

하드코딩된 ~/forza_ws/race_stack 경로를 제거하여
평가 컨테이너(/home/misys/...) 등 어떤 환경에서도 지도와 설정 파일을 자동으로 찾는다.
"""
import os
from pathlib import Path


def resolve_maps_dir() -> Path:
    """ForzaETH stack_master/maps 디렉토리 경로를 자동 탐색한다."""
    env_dir = os.environ.get("FORZA_MAPS_DIR")
    if env_dir and Path(env_dir).is_dir():
        return Path(env_dir)

    try:
        from ament_index_python.packages import get_package_share_directory
        pkg_maps = Path(get_package_share_directory("stack_master")) / "maps"
        if pkg_maps.is_dir():
            return pkg_maps
    except Exception:
        pass

    candidates = [
        Path.home() / "race_stack" / "stack_master" / "maps",
        Path.home() / "forza_ws" / "race_stack" / "stack_master" / "maps",
        Path("/home/misys/race_stack/stack_master/maps"),
        Path("/home/misys/forza_ws/race_stack/stack_master/maps"),
    ]
    for path in candidates:
        if path.is_dir():
            return path

    raise FileNotFoundError(
        "stack_master/maps 디렉토리를 찾을 수 없습니다. "
        "FORZA_MAPS_DIR 환경변수를 설정하거나 ROS 2 워크스페이스를 source 하세요."
    )


def resolve_project_root() -> Path:
    """현재 프로젝트의 루트 디렉토리를 반환한다."""
    return Path(__file__).resolve().parent.parent
