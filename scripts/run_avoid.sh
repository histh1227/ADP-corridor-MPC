#!/bin/bash
# 경로 자동 탐색 및 ROS 2 자동 source가 적용된 통합 실행 스크립트
# 사용법: ./scripts/run_avoid.sh [장애물 수] [시드(-1이면 무작위)] [맵이름]

NOBS=${1:-4}
SEED=${2:--1}
MAP=${3:-berlin}
SPEC="localization.lidar_localizer:LidarLocDriver"
KW="{\"init\": \"start\", \"forbid_gt\": true, \"inner\": \"drivers.unified_corridor_mpc:UnifiedCorridorMPC\", \"profile\": \"profiles/berlin_unified.json\", \"map_name\": \"$MAP\"}"

HERE=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$HERE:$PYTHONPATH"

# 1. ForzaETH 워크스페이스 자동 탐색
STACK=""
for cand in "$FORZA_STACK_DIR" "/home/misys/race_stack" "$HOME/race_stack" "$HOME/forza_ws/race_stack"; do
  if [ -n "$cand" ] && [ -f "$cand/install/setup.bash" ]; then
    STACK="$cand"
    break
  fi
done
if [ -z "$STACK" ]; then
  echo "[오류] ForzaETH race_stack(install/setup.bash)을 찾을 수 없습니다."
  exit 1
fi

MAPS="$STACK/stack_master/maps"
export FORZA_MAPS_DIR="$MAPS"
[ -f "$MAPS/$MAP/global_waypoints.json" ] || { echo "맵 파일 없음: $MAPS/$MAP/global_waypoints.json"; exit 1; }

# 2. ROS 2 자동 source
source /opt/ros/humble/setup.bash
source "$STACK/install/setup.bash"
export PYTHONPATH="$HERE:$PYTHONPATH"

read SX SY QZ QW < <(python3 -c "
import json, math
w = json.load(open('$MAPS/$MAP/global_waypoints.json'))['global_traj_wpnts_iqp']['wpnts'][0]
print(w['x_m'], w['y_m'], math.sin(w['psi_rad'] / 2), math.cos(w['psi_rad'] / 2))
")

LOG=~/race_logs
mkdir -p "$LOG"

GUARD="ros2 launch stack_master|avoid_node.py|controller_manager_cpp|sqp_avoidance_node"
if pgrep -u "$(id -u)" -f "$GUARD" > /dev/null; then
  echo "이미 실행 중인 스택/노드가 있습니다. 먼저 종료하세요:"
  pgrep -u "$(id -u)" -af "$GUARD"
  exit 1
fi

set -m
cleanup() {
  trap - EXIT INT TERM
  local groups=$(jobs -p)
  for g in $groups; do kill -INT -- -$g 2>/dev/null; done
  for i in $(seq 15); do
    alive=0; for g in $groups; do kill -0 -- -$g 2>/dev/null && alive=1; done
    [ $alive -eq 0 ] && break; sleep 1
  done
  for g in $groups; do kill -TERM -- -$g 2>/dev/null; done; sleep 2
  for g in $groups; do kill -KILL -- -$g 2>/dev/null; done
  exit 0
}
trap cleanup EXIT INT TERM

wait_for() {
  local t=0
  until grep -qE "$1" "$2" 2>/dev/null; do
    sleep 1; t=$((t + 1))
    if [ $t -ge $3 ] || ! kill -0 $4 2>/dev/null; then
      echo "실행 실패: '$1' 미감지 ($2, ${t}s 경과)"
      tail -20 "$2"
      exit 1
    fi
  done
}

cd "$STACK"
ros2 launch stack_master base_system_launch.xml map_name:=$MAP sim:=true pose_source:=gt > "$LOG/av_base.log" 2>&1 &
wait_for "Managed nodes are active" "$LOG/av_base.log" 120 $!
echo "[1/3] 시뮬레이터 준비 완료 ($MAP)"

if ros2 pkg prefix race_manager > /dev/null 2>&1; then
  ros2 launch race_manager race_manager_launch.xml map_name:=$MAP seed:=$SEED num_obstacles:=$NOBS > "$LOG/av_race.log" 2>&1 &
  wait_for "lap 0 \[" "$LOG/av_race.log" 60 $!
  echo "[2/3] race_manager 준비 완료 (장애물 $NOBS개, 시드 $SEED)"
else
  echo "[2/3] race_manager 없음: 장애물 없이 주행"
  : > "$LOG/av_race.log"
fi

ros2 topic pub --once -w 2 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped   "{header: {frame_id: map}, pose: {pose: {position: {x: $SX, y: $SY, z: 0.0}, orientation: {z: $QZ, w: $QW}}, covariance: [0.25,0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.068]}}" > /dev/null

(cd "$HERE" && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   exec python3 "$HERE/nodes/avoid_node.py" "$SPEC" "$KW" 50 > "$LOG/av_driver.log" 2>&1) &
wait_for "avoid driver ready|Traceback" "$LOG/av_driver.log" 120 $!
grep -q Traceback "$LOG/av_driver.log" && { cat "$LOG/av_driver.log"; exit 1; }

echo "[3/3] 주행 시작: $SPEC (종료: Ctrl-C, 로그: $LOG/av_driver.log)"
tail -n 0 -f "$LOG/av_race.log" | grep --line-buffered -E "  lap [0-9]+  best|collision #|obstacles —" &
wait
