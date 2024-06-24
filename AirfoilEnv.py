import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import io
from PIL import Image
import torch
import cv2
from simulation import run_simulation
from OPENFOAM_MAKER import make_block_mesh_dict
from utils import bezier_curve
from scipy.interpolate import interp1d


class CustomAirfoilEnv:
    def __init__(self, num_points, angle_of_attack):
        self.num_points = num_points
        self.angle_of_attack = angle_of_attack
        self._initial_circles = [((0.02, 0), 0.02), ((1 - 0.02, 0), 0.02)]
        # 초기 상태 설정
        self.circles = self._initial_circles.copy()
        self.points, self.state, _ = self.get_airfoil(self.circles)
        self.prev_lift_drag_ratio = 4.5

    def reset(self):
        self.circles = self._initial_circles.copy()
        self.points, self.state, _ = self.get_airfoil(self.circles)
        self.prev_lift_drag_ratio = 4.5
        return self.get_state()

    def step(self, action, t=None):
        self.circles.append(((action[0], action[1]), action[2]))  # add circle with x, r
        points, state, img = self.get_airfoil(self.circles, t=t)
        self.points = points
        make_block_mesh_dict(
            points[:, 0], points[:, 1], angle_of_attack=self.angle_of_attack
        )
        Cd, Cl = run_simulation()
        lift_drag_ratio = self.calculate_reward(Cd, Cl)

        improvement = lift_drag_ratio - self.prev_lift_drag_ratio
        reward = improvement
        if improvement > 0:
            reward += 1
        self.prev_lift_drag_ratio = lift_drag_ratio
        self.state = state

        return state, reward, lift_drag_ratio, img

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

    def get_airfoil(self, circles, t=None, save_path="airfoil.png"):
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

        num_points = len(interpolated_points)
        airfoil = bezier_curve(interpolated_points, num=num_points)

        # 단일 Figure 객체와 Axes 객체 생성
        fig, ax = plt.subplots(
            figsize=(6.8, 4.8)
        )  # 인치 단위로 크기 설정 (6.8*50, 4.8*50 = 340, 240)
        ax.fill(airfoil[:, 0], airfoil[:, 1], "k")
        ax.set_aspect("equal")
        ax.set_xlim(-0.2, 1.5)
        ax.set_ylim(-0.6, 0.6)
        ax.axis("off")

        if save_path is not None:
            fig.savefig(
                save_path, format="png", dpi=50, bbox_inches="tight", pad_inches=0
            )

        # 저장한 이미지를 다시 불러오기
        img = Image.open(save_path)

        # 메모리에 이미지 저장
        buf = io.BytesIO()
        fig.savefig(
            buf, format="png", dpi=50, bbox_inches="tight", pad_inches=0
        )  # 50 DPI로 저장
        buf.seek(0)

        sdf = self.apply_sdf(buf)

        sdf_tensor = torch.tensor(sdf).unsqueeze(0).float()
        resized_sdf_tensor = torch.nn.functional.interpolate(
            sdf_tensor.unsqueeze(0),
            size=(240, 340),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # 이미지 크기를 조정하지 않고 바로 반환
        return interpolated_points, resized_sdf_tensor, img

    def apply_sdf(self, buf):
        image = Image.open(buf).convert("L")
        image_array = np.array(image)

        _, binary_img = cv2.threshold(image_array, 127, 255, cv2.THRESH_BINARY)

        dist_outside = cv2.distanceTransform(255 - binary_img, cv2.DIST_L2, 5)
        dist_inside = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
        sdf = dist_inside - dist_outside

        sdf /= 100

        return sdf

    def plot_airfoil(self, hull_points, interpolated_points, t=None):
        plt.figure(figsize=(5, 10))  # 새로운 그림 생성
        plt.gca().set_aspect("equal", adjustable="box")  # 비율을 유지합니다.

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
        plt.savefig("airfoil.png")  # 파일 이름에 인덱스 추가
        plt.close("all")  # 그림 닫기

    def interpolate_linear_functions(self, hull_points):
        """
        Convex Hull 점을 사용하여 선형 함수를 보간합니다.
        """

        # Normalize the hull points
        x_min = np.min(hull_points[:, 0])
        x_argmin = np.argmin(hull_points[:, 0])
        y_standard = hull_points[x_argmin, 1]
        hull_points -= [x_min, y_standard]

        # Sort points by x values
        hull_points = hull_points[hull_points[:, 0].argsort()]

        # Separate upper and lower points
        upper_points = hull_points[hull_points[:, 1] >= 0]
        lower_points = hull_points[hull_points[:, 1] < 0]

        # Sampling points
        sampling_points = np.array(
            [
                1,
                0.95,
                0.9,
                0.8,
                0.7,
                0.6,
                0.5,
                0.4,
                0.3,
                0.25,
                0.2,
                0.15,
                0.1,
                0.075,
                0.05,
                0.025,
                0.0125,
                0,
            ]
        )

        # Interpolate upper points
        upper_x = upper_points[:, 0]
        upper_y = upper_points[:, 1]
        upper_interp = interp1d(
            upper_x, upper_y, kind="linear", fill_value="extrapolate"
        )
        upper_sampling_y = upper_interp(sampling_points)

        # Interpolate lower points
        lower_x = lower_points[:, 0]
        lower_y = lower_points[:, 1]
        lower_interp = interp1d(
            lower_x, lower_y, kind="linear", fill_value="extrapolate"
        )
        lower_sampling_y = lower_interp(sampling_points[::-1])  # Reverse for lower

        # Combine upper and lower sampling points
        upper_sampling_points = np.vstack((sampling_points, upper_sampling_y)).T
        lower_sampling_points = np.vstack((sampling_points[::-1], lower_sampling_y)).T

        # Combine upper and lower points into a single array
        sampled_points = np.vstack((upper_sampling_points, lower_sampling_points))
        return sampled_points


def make_env(num_points=80, angle_of_attack=5.0):
    return CustomAirfoilEnv(num_points=num_points, angle_of_attack=angle_of_attack)
