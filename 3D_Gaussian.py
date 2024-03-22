from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt


def ellipse_points(center, covariance, n=100):
    """타원 경계에 대한 점들을 계산하고 반환합니다."""
    t = np.linspace(0, 2 * np.pi, n)
    eig_vals, eig_vecs = np.linalg.eigh(covariance)
    # 타원의 주 축에 따라 점들을 생성합니다.
    ellipse = np.array(
        [np.sqrt(eig_vals[0]) * np.cos(t), np.sqrt(eig_vals[1]) * np.sin(t)]
    )
    # 고유 벡터에 의한 회전을 적용합니다.
    ellipse = eig_vecs @ ellipse
    # 중심을 더해 위치를 조정합니다.
    ellipse = ellipse.T + center
    return ellipse


# 타원 그리기 및 중간 타원 보간을 위한 설정
centers = [np.array([0, 0]), np.array([2, 2])]
covariances = [np.array([[1, 0.8], [0.8, 1]]), np.array([[1, -0.5], [-0.5, 1]])]

# 다시 시도하여 모든 타원의 경계 점들을 수집하고 컨벡스 헐을 계산 및 시각화합니다.
fig, ax = plt.subplots()

all_points = np.vstack(
    [ellipse_points(center, cov, n=100) for center, cov in zip(centers, covariances)]
)

hull = ConvexHull(all_points)
for simplex in hull.simplices:
    plt.plot(all_points[simplex, 0], all_points[simplex, 1], "k-")

# 그래프 범위 조정 및 표시
ax.set_xlim(-3, 5)
ax.set_ylim(-3, 5)
plt.show()
