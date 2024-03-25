import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# 원의 경계에 있는 점들을 생성하는 함수
def generate_circle_points(center, radius, num_points=100):
    return np.array([
        [center[0] + np.cos(2 * np.pi / num_points * x) * radius, 
         center[1] + np.sin(2 * np.pi / num_points * x) * radius] 
        for x in range(num_points)])

# 원들의 중심과 반지름을 저장하는 리스트
circles = [((-1, 0), 0.5), ((0, 0), 1), ((1, 0), 0.5),((2, 0), 1)]

# 모든 원들의 점들을 생성하고 합치는 과정
all_points = np.concatenate([generate_circle_points(center, radius) for center, radius in circles])

# Convex hull 계산
hull = ConvexHull(all_points)

# Convex hull 그리기
plt.figure(figsize=(8, 6))
plt.plot(all_points[:,0], all_points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(all_points[simplex, 0], all_points[simplex, 1], 'k-')

# 그래프 조정
plt.xlim(np.min(all_points[:,0]) - 1, np.max(all_points[:,0]) + 1)
plt.ylim(np.min(all_points[:,1]) - 1, np.max(all_points[:,1]) + 1)
plt.show()
