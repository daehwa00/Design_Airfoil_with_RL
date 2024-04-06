import numpy as np
from xfoil import XFoil
from xfoil.model import Airfoil
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from torchvision import transforms
import torch

class CustomAirfoilEnv:
    def __init__(self):
        self.xfoil = XFoil()
        self.xfoil.Re = 1e6
        self.xfoil.max_iter = 100  # 최대 반복 횟수
        self.circles = [((0, 0), 0.05), ((1, 0), 0.002)]
        self.points, self.state = self.get_airfoil(self.circles)  # state는 점을 제공 shape=(N, 2)
        self.xfoil.airfoil = Airfoil(self.points[:, 0], self.points[:, 1])

    def reset(self):
        self.circles = [((0, 0), 0.05), ((1, 0), 0.001)]
        self.points, self.state = self.get_airfoil(self.circles)
        self.xfoil.airfoil = Airfoil(self.points[:, 0], self.points[:, 1])
        return self.get_state()

    def step(self, action, t=None):
        self.circles.append(((action[0], 0), action[1]))  # add circle
        points, state = self.get_airfoil(self.circles, t=t)
        self.xfoil.airfoil = Airfoil(points[:, 0], points[:, 1])
        cl, cd, cm, cp = self.xfoil.a(5)    # angle of attack is 5 degrees
        """
        Cl : 양력 계누는 음수일 수 있다.
        자동차 레이싱에서는 의도적으로 음수가 되어 차량을 도로에 더 단단히 밀착시킨다.
        Cd : 유체의 흐름에 대항하는 물체의 저항
        """
        # reward = cl / (cd + 1e-5)
        reward = cl

        if reward == 0 or np.isnan(reward):
            reward = -1

        next_state = state
        self.state = next_state

        return next_state, reward

    def get_state(self):
        return self.state

    def generate_all_circle_points(self, circles, num_points=100):
        """
        Generate all points of circles.
        """
        def generate_circle_points(center, radius, num_points=100):
            return np.array(
                [
                    [
                        center[0] + np.cos(2 * np.pi / num_points * x) * radius,
                        center[1] + np.sin(2 * np.pi / num_points * x) * radius,
                    ]
                    for x in range(num_points)
                ]
            )
        all_points = np.concatenate(
            [
                generate_circle_points(center, radius, num_points)
                for center, radius in circles
            ]
        )

        return all_points


    def get_airfoil(self, circles, N=100, plot=False, t=None):
        """
        Generate airfoil points using linear interpolation of convex hull points.
        """
        all_points = self.generate_all_circle_points(circles)
        all_points = all_points[all_points[:, 0] <= 1]
        hull = ConvexHull(all_points)
        hull_points = all_points[hull.vertices]
        interpolated_points = self.interpolate_linear_functions(hull_points, N=N)

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
            plt.text(0.05, 0.95, f"Image #{t + 1}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Linear Interpolation of Convex Hull Points")
        plt.savefig('airfoil.png')

        cs_upper = CubicSpline(np.concatenate(([0],np.flip(interpolated_points[0,:100]))), np.concatenate(([0],np.flip(interpolated_points[1,:100]))))
        cs_lower = CubicSpline(np.concatenate(([0],interpolated_points[0,100:])), np.concatenate(([0],interpolated_points[1,100:])))
        x_fine_upper = np.linspace(0, 1, 10000)
        x_fine_lower = np.linspace(0, 1, 10000)

        # 그래프 그리기
        fig = Figure(figsize=(10, 5))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(x_fine_upper, cs_upper(x_fine_upper), color='black')
        ax.plot(x_fine_lower, cs_lower(x_fine_lower), color='black')
        ax.axis('off')
        ax.axis('equal')

        # 메모리에 이미지 저장
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=50)
        buf.seek(0)

        image = Image.open(buf).convert('L')
        tensor = 1 - transforms.ToTensor()(image)
        buf.close()

        return interpolated_points.T, tensor

    def interpolate_linear_functions(self, hull_points, N=100):
        x_min = np.min(hull_points[:, 0])
        x_argmin = np.argmin(hull_points[:, 0])
        y_standard = hull_points[x_argmin, 1]
        hull_points -= [x_min, y_standard]

        N_front = int(0.3 * N)
        N_back = N - N_front

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
        upper_back_x_values = np.linspace(0.1, 1, N_back, endpoint=True)
        upper_x_values = np.concatenate((upper_front_x_values, upper_back_x_values))
        upper_y_values = np.zeros(N)

        upper_current_segment = 0
        for i in range(N):
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
        lower_y_values = np.zeros(N)

        lower_current_segment = 0
        for i in range(N):
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

        x_values = np.concatenate((upper_x_values, lower_x_values))
        y_values = np.concatenate((upper_y_values, lower_y_values))
        values = np.vstack((x_values, y_values))

        return values
def make_env():
    return CustomAirfoilEnv()
