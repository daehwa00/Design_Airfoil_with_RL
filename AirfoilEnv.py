import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
import torch
import cv2
import bezier
from blockMeshDictMaker import make_block_mesh_dict
from simulation import run_simulation


class CustomAirfoilEnv:
    def __init__(self, num_points=80):
        self.num_points = num_points
        self._initial_circles = [((0.03, 0), 0.03), ((1-0.03, 0), 0.03)]
        # 초기 상태 설정
        self.circles = self._initial_circles.copy()
        self.points, self.state = self.get_airfoil(
            self.circles
        ) 

    def reset(self):
        self.circles = self._initial_circles.copy()
        self.points, self.state = self.get_airfoil(self.circles)
        return self.get_state()

    def step(self, action, t=None):
        self.circles.append(((action[0], 0), action[1]))  # add circle with x, r
        points, state = self.get_airfoil(self.circles, t=t)
        self.points = points
        Cd, Cl = run_simulation()
        reward = self.calculate_reward(Cd, Cl)
        next_state = state
        self.state = next_state

        return next_state, reward

    def calculate_reward(self, Cd, Cl):
        # 양항비
        lift_drag_ratio = Cl / Cd
        return lift_drag_ratio

    def get_state(self):
        """
        현재 상태(Airfoil)을 반환합니다.
        """
        return self.state

    def generate_all_circle_points(self, circles):
        """
        원의 중심과 반지름을 사용하여 모든 점을 생성합니다.
        """

        def generate_circle_points(center, radius):
            return np.array(
                [
                    [
                        center[0] + np.cos(2 * np.pi / self.num_points * x) * radius,
                        center[1] + np.sin(2 * np.pi / self.num_points * x) * radius,
                    ]
                    for x in range(self.num_points)
                ]
            )

        all_points = np.concatenate(
            [generate_circle_points(center, radius) for center, radius in circles]
        )

        return all_points

    def get_airfoil(self, circles, t=None):
        """
        주어진 원들을 사용하여 airfoil을 생성합니다.
        """
        all_points = self.generate_all_circle_points(circles)
        all_points = all_points[
            all_points[:, 0] <= 1
        ]  # x 좌표가 1보다 작거나 같은 점만 유지
        hull = ConvexHull(all_points)
        hull_points = all_points[hull.vertices]
        interpolated_points = self.interpolate_linear_functions(hull_points)

        make_block_mesh_dict(
            interpolated_points[0], interpolated_points[1]
        )  # blockMeshDict 생성, controlDict는 고정

        self.plot_airfoil(hull_points, interpolated_points, t)

        # Cubic Spline을 사용하여 보간된 점을 연결
        cs_upper = bezier.Curve(
            [
                (np.flip(interpolated_points[0, : self.num_points])),
                (np.flip(interpolated_points[1, : self.num_points])),
            ],
            degree=self.num_points - 1,
        )
        cs_lower = bezier.Curve(
            [
                (interpolated_points[0, self.num_points :]),
                (interpolated_points[1, self.num_points :]),
            ],
            degree=self.num_points,
        )
        x_fine_upper = np.linspace(0, 1, 10000)
        x_fine_lower = np.linspace(0, 1, 10000)

        cs_upper = cs_upper.evaluate_multi(x_fine_upper)
        cs_lower = cs_lower.evaluate_multi(x_fine_lower)

        fig = Figure(figsize=(10, 5))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(x_fine_upper, cs_upper[1], color="black")
        ax.plot(x_fine_lower, cs_lower[1], color="black")
        ax.fill_between(x_fine_upper, cs_upper[1], cs_lower[1], color="black")
        ax.axis("off")
        ax.axis("equal")

        # 메모리에 이미지 저장
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=50)
        buf.seek(0)

        sdf = self.apply_sdf(buf)

        return interpolated_points.T, torch.tensor(sdf).unsqueeze(0).float()

    def apply_sdf(self, buf):
        image = Image.open(buf).convert("L")
        image.save("airfoil_image.png")
        _, binary_img = cv2.threshold(
            np.array(image), 127, 255, cv2.THRESH_BINARY
        )  # 이진 이미지로 변환
        dist_outside = cv2.distanceTransform(255 - binary_img, cv2.DIST_L2, 5)
        dist_inside = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
        sdf = dist_inside - dist_outside
        sdf = sdf / np.max(np.abs(sdf))
        return sdf

    def plot_airfoil(self, hull_points, interpolated_points, t=None):
        plt.figure(figsize=(10, 5))
        # xlim과 ylim을 같게 설정하여 비율을 유지합니다.
        plt.gca().set_aspect("equal")
        plt.plot(
            np.array(hull_points[:, 0]),
            np.array(hull_points[:, 1]),
            "o-",
            label="Hull Points",
        )
        plt.plot(
            np.array(interpolated_points[0]),
            np.array(interpolated_points[1]),
            ".r",
            label="Interpolated Points",
        )
        if t is not None:  # t가 주어진 경우, 이미지에 텍스트 추가
            plt.text(
                0.05,
                0.95,
                f"Image #{t + 1}",
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment="top",
            )
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Linear Interpolation of Convex Hull Points")
        plt.savefig("airfoil.png")

    def interpolate_linear_functions(self, hull_points):
        """
        Convex Hull 점을 사용하여 선형 함수를 보간합니다.
        """

        N_front = int(0.3 * self.num_points)
        N_back = self.num_points - N_front

        # x 좌표의 최대값으로 모든 x 좌표를 정규화
        x_max = np.max(hull_points[:, 0])
        hull_points[:, 0] /= x_max
        hull_points[:, 1] /= x_max  # y 좌표도 x 최대값으로 나누어 비율 유지

        hull_points = np.vstack([hull_points, hull_points[0]])  # 경로 닫기

        # x 좌표를 기준으로 정렬 (시계 방향 또는 반시계 방향 보장)
        hull_points = hull_points[np.argsort(hull_points[:, 0])]

        # UPPER
        upper_hull_points = hull_points[hull_points[:, 1] >= 0]
        # 각 선분에 대한 x 및 y의 기울기 계산
        upper_dx = np.diff(upper_hull_points[:, 0])
        upper_dy = np.diff(upper_hull_points[:, 1])
        upper_slopes = upper_dy / upper_dx
        upper_intercepts = (
            upper_hull_points[:-1, 1] - upper_slopes * upper_hull_points[:-1, 0]
        )

        # N+1을 사용하고 endpoint=False를 추가합니다.
        upper_front_x_values = np.geomspace(
            0.0001, 0.1, N_front, endpoint=False
        )  # 0 대신 최소값으로 시작

        # UPPER
        upper_back_x_values = np.linspace(0.1, 1, N_back, endpoint=True)
        upper_x_values = np.concatenate((upper_front_x_values, upper_back_x_values))
        upper_y_values = np.zeros(self.num_points)

        upper_current_segment = 0
        for i in range(self.num_points):
            upper_x = upper_x_values[i]
            while (
                upper_current_segment < len(upper_slopes) - 1
                and upper_x > upper_hull_points[upper_current_segment + 1, 0]
            ):
                upper_current_segment += 1
            upper_y_values[i] = (
                upper_slopes[upper_current_segment] * upper_x
                + upper_intercepts[upper_current_segment]
            )

        upper_x_values = np.flip(upper_x_values)  # x 좌표를 다시 뒤집습니다.
        upper_y_values = np.flip(upper_y_values)
        # LOWER
        lower_hull_points = hull_points[hull_points[:, 1] <= 0]

        lower_dx = np.diff(lower_hull_points[:, 0])
        lower_dy = np.diff(lower_hull_points[:, 1])
        lower_slopes = lower_dy / lower_dx
        lower_intercepts = (
            lower_hull_points[:-1, 1] - lower_slopes * lower_hull_points[:-1, 0]
        )

        lower_front_x_values = np.geomspace(
            0.0001, 0.1, N_front, endpoint=False
        )  # 0 대신 최소값으로 시작
        lower_back_x_values = np.linspace(0.1, 1, N_back, endpoint=True)
        lower_x_values = np.concatenate((lower_front_x_values, lower_back_x_values))
        lower_y_values = np.zeros(self.num_points)

        lower_current_segment = 0
        for i in range(self.num_points):
            lower_x = lower_x_values[i]
            while (
                lower_current_segment < len(lower_slopes) - 1
                and lower_x > lower_hull_points[lower_current_segment + 1, 0]
            ):
                lower_current_segment += 1
            lower_y_values[i] = (
                lower_slopes[lower_current_segment] * lower_x
                + lower_intercepts[lower_current_segment]
            )

        x_values = np.concatenate((upper_x_values, [0], lower_x_values))
        y_values = np.concatenate((upper_y_values, [0], lower_y_values))
        values = np.vstack((x_values, y_values))

        return values


def make_env(num_points=80):
    return CustomAirfoilEnv(num_points=num_points)
