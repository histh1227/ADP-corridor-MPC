#!/bin/bash
set -e

HERE=$(cd "$(dirname "$0")/.." && pwd)
echo "[1/3] Python 패키지 설치..."
pip install --user -r "$HERE/requirements.txt"

echo "[2/3] 실행 스크립트 권한 설정..."
chmod +x "$HERE/scripts/run_avoid.sh"

echo "[3/3] ROS 2 환경 로드 및 Numba 커널 사전 컴파일..."
source /opt/ros/humble/setup.bash || true
for cand in "$FORZA_STACK_DIR" "/home/misys/race_stack" "$HOME/race_stack" "$HOME/forza_ws/race_stack"; do
  if [ -n "$cand" ] && [ -f "$cand/install/setup.bash" ]; then
    source "$cand/install/setup.bash"
    export FORZA_MAPS_DIR="$cand/stack_master/maps"
    break
  fi
done
python3 "$HERE/scripts/warmup_numba.py"

echo "설치 및 사전 컴파일 완료! 이제 ./scripts/run_avoid.sh 로 바로 실행할 수 있습니다."
