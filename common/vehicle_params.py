"""차량 및 LiDAR 물리 상수 (sim_params.yaml 기본값과 일치).

Localization, Perception, Planning, Control 모든 모듈이 이 단일 파일의 상수를 공유한다.
"""
import numpy as np

# 시뮬레이터 single-track 모델 상수 (sim_params.yaml)
MU_ = 1.0489
CSF_ = 4.718
CSR_ = 5.4562
LF_ = 0.15875
LR_ = 0.17145
HCG_ = 0.074
G_ = 9.81
M_ = 3.47
I_ = 0.04712
WB = LF_ + LR_

STEER_MAX = 0.4189
STEER_RATE = 3.2
KP_ACC = 10.0 * 9.51 / 20.0
KP_BRK = 10.0 * 9.51 / 5.0
A_MAX = 9.51
V_SWITCH = 7.319

# LiDAR 및 충돌 박스 상수
LIDAR_X = 0.27
BOX_HW = 0.1016        # 충돌 판정 박스 반폭
BOX_FRONT = 0.435      # CG 기준 충돌 박스 전면 거리
ANG_MIN = -2.35
ANG_INC = 4.7 / 1079
NB = 1080
K_US0 = 0.002787

# 차체 검사 기준점 (CG 좌표계, m)
BODY_PTS = (
    (-0.30, 0.12), (-0.30, -0.12), (-0.30, 0.0),
    (-0.15, 0.12), (-0.15, -0.12),
    (0.0, 0.12), (0.0, -0.12),
    (0.105, 0.11), (0.105, -0.11),
    (0.2, 0.11), (0.2, -0.11),
    (0.3, 0.11), (0.3, -0.11),
    (BOX_FRONT, 0.11), (BOX_FRONT, -0.11), (BOX_FRONT, 0.0),
)
FRONT_PTS = tuple(p for p in BODY_PTS if p[0] >= 0.0)


def k_us(a, c=1.0):
    """종방향 가속도 a(하중 이동)와 타이어 강성 배율 c에 따른 언더스티어 구배."""
    a = np.minimum(np.maximum(a, -A_MAX), A_MAX)
    return (
        LR_ / (MU_ * CSF_ * (G_ * LR_ - a * HCG_))
        - LF_ / (MU_ * CSR_ * (G_ * LF_ + a * HCG_))
    ) / c


def wrap_angle(a):
    """각도를 [-pi, pi) 범위로 정규화한다."""
    return (a + np.pi) % (2 * np.pi) - np.pi
