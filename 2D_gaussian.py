import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull


def draw_ellipse(center, covariance, ax, n_std=1.0, **kwargs):
    """주어진 중심과 공분산 행렬로 타원을 그립니다."""
    eig_vals, eig_vecs = np.linalg.eigh(covariance)
    angle = np.degrees(np.arctan2(*eig_vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eig_vals)
    # 여기서 angle을 키워드 인자로 전달합니다.
    ellipse = Ellipse(center, width, height, angle=angle, **kwargs)
    ax.add_patch(ellipse)


def ellipse_points(center, covariance, n=100):
    """타원 경계에 대한 점들을 계산하고 반환합니다."""
    t = np.linspace(0, 2 * np.pi, n)
    eig_vals, eig_vecs = np.linalg.eigh(covariance)
    ellipse_rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    ellipse = np.sqrt(eig_vals)[:, np.newaxis] * ellipse_rot
    ellipse = eig_vecs @ ellipse + center[:, np.newaxis]
    return ellipse.T


# 타원 그리기 및 중간 타원 보간을 위한 설정
centers = [np.array([0, 0]), np.array([2, 2])]
covariances = [np.array([[1, 0.8], [0.8, 1]]), np.array([[1, -0.5], [-0.5, 1]])]

# 시각화 준비
fig, ax = plt.subplots()

# 모든 타원의 경계 점들을 수집합니다.
all_points = np.vstack(
    [ellipse_points(center, cov, n=100) for center, cov in zip(centers, covariances)]
)

# 각 가우시안 분포에 대한 타원 그리기
for center, cov in zip(centers, covariances):
    draw_ellipse(center, cov, ax, n_std=1.0, edgecolor="blue", facecolor="none")


# 컨벡스 헐 계산 및 시각화
hull = ConvexHull(all_points)
for simplex in hull.simplices:
    plt.plot(all_points[simplex, 0], all_points[simplex, 1], "k-")

# 점들 시각화
plt.plot(hull, "o", markersize=3)

# 그래프 범위 조정 및 표시
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
plt.show()
