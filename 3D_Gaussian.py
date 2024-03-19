import numpy as np
import pyvista as pv


def generate_3d_gaussian_mesh(center, covariance_matrix, grid_size=50):
    # 격자 범위 정의
    x, y, z = np.mgrid[
        -3 : 3 : complex(grid_size),
        -3 : 3 : complex(grid_size),
        -3 : 3 : complex(grid_size),
    ]
    grid = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

    # 공분산 행렬의 역행렬 및 결정자 계산
    inv_cov = np.linalg.inv(covariance_matrix)
    det_cov = np.linalg.det(covariance_matrix)

    # 가우시안 분포 계산
    n = np.sqrt((2 * np.pi) ** 3 * det_cov)
    diff = grid - center
    exp_part = np.exp(-0.5 * (np.einsum("ij,jk,ik->i", diff, inv_cov, diff)))
    pdf = exp_part / n

    # 메시 데이터 생성
    points = grid.reshape((grid_size, grid_size, grid_size, 3))
    pdf = pdf.reshape((grid_size, grid_size, grid_size))

    # PyVista 그리드 생성
    grid = pv.ImageData()
    grid.dimensions = np.array(pdf.shape)
    grid.spacing = [6 / (grid_size - 1)] * 3
    grid.origin = center - 3  # 격자 시작점 조정
    grid.point_data["scalars"] = pdf.flatten(order="F")

    # 등고선(메시) 생성
    contours = grid.contour(isosurfaces=4, scalars="scalars")
    return contours


# 중심, 공분산 행렬 및 격자 크기 설정
center = np.array([0, 0, 0])
covariance_matrix = np.array([[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]])

# 메시 생성 및 시각화
mesh = generate_3d_gaussian_mesh(center, covariance_matrix, grid_size=50)
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="white", show_edges=True)
plotter.show()
