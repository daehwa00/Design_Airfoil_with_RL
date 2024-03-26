import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d

def generate_all_circle_points(circles, num_points=100):
    """
    여러 원들에 대한 점들을 생성하고 합칩니다.
    
    :param circles: 각 원의 (중심, 반지름) 튜플을 포함하는 리스트입니다.
    :param num_points: 각 원을 대표하는 점의 수입니다 (기본값: 100).
    :return: 생성된 모든 원들의 점들을 합친 numpy 배열입니다.
    """
    def generate_circle_points(center, radius, num_points=100):
        return np.array([
            [center[0] + np.cos(2 * np.pi / num_points * x) * radius, 
             center[1] + np.sin(2 * np.pi / num_points * x) * radius] 
            for x in range(num_points)])
    
    # 모든 원들의 점들을 생성하고 합칩니다.
    all_points = np.concatenate([generate_circle_points(center, radius, num_points) for center, radius in circles])

    return all_points

    


def interpolate_linear_functions(hull_points, N=100):
    hull_points = np.vstack([hull_points, hull_points[0]])  # 경로 닫기
    # x 좌표를 기준으로 정렬 (시계 방향 또는 반시계 방향 보장)
    hull_points = hull_points[np.argsort(hull_points[:, 0])]

    # 각 선분에 대한 x 및 y의 기울기 계산
    dx = np.diff(hull_points[:, 0])
    dy = np.diff(hull_points[:, 1])
    slopes = dy / dx
    intercepts = hull_points[:-1, 1] - slopes * hull_points[:-1, 0]

    # 각 선분의 길이 계산
    segment_lengths = np.sqrt(dx**2 + dy**2)
    total_length = np.sum(segment_lengths)

    # N개의 x값을 균등하게 분배
    x_values = np.linspace(hull_points[0, 0], hull_points[-1, 0], N)
    y_values = np.zeros(N)

    # 각 x값에 대응하는 y값 계산
    current_segment = 0
    for i in range(N):
        x = x_values[i]
        # 해당 x값이 현재 선분 범위를 벗어난 경우, 다음 선분으로 이동
        while current_segment < len(slopes) - 1 and x > hull_points[current_segment + 1, 0]:
            current_segment += 1
        # 선형 보간으로 y값 계산
        y_values[i] = slopes[current_segment] * x + intercepts[current_segment]

    return np.vstack((x_values, y_values)).T

# 에어포일 점들 생성 (예시 점들 사용)
circles = [((-1, 0), 0.5), ((0, 0), 1), ((1, 0), 0.5), ((2, 0), 1)]
all_points = generate_all_circle_points(circles)
hull = ConvexHull(all_points)
hull_points = all_points[hull.vertices]


# 보간된 점들 계산
interpolated_points = interpolate_linear_functions(hull_points, N=100)

# 시각화
plt.figure(figsize=(10, 5))
plt.plot(hull_points[:, 0], hull_points[:, 1], 'o-', label='Hull Points')
plt.plot(interpolated_points[:, 0], interpolated_points[:, 1], '.r', label='Interpolated Points')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Interpolation of Convex Hull Points')
plt.show()