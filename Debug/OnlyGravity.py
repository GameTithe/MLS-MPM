import numpy as np
 
pos = np.array([31.5, 31.5, 0])  # 초기 위치 (x, y, z)
vel = np.array([0.0, 0.0, 0.0])  # 초기 속도 (0, 0, 0)
gravity = np.array([0.0, -9.8, 0.0])  # 중력 가속도 (0, -9.8, 0)
dt = 1e-3  # 시간 간격 (초)
time_end = 1.0  # 시뮬레이션 지속 시간 (초)

# 시뮬레이션 루프
time = 0.0
while time < time_end:
    vel += gravity * dt  # 속도 업데이트 (v = v0 + a*t)
    pos += vel * dt  # 위치 업데이트 (x = x0 + v*t)
    
    print(f"t = {time:.3f}s, pos = {pos}")  # 좌표 출력
    
    time += dt
