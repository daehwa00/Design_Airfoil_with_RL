import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    """모든 난수 생성기의 시드를 고정합니다."""
    torch.manual_seed(seed)  # PyTorch를 위한 시드 고정
    torch.cuda.manual_seed(seed)  # CUDA를 위한 시드 고정
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU를 위한 시드 고정
    np.random.seed(seed)  # NumPy를 위한 시드 고정
    random.seed(seed)  # Python 내장 random 모듈을 위한 시드 고정
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python 해시 생성에 사용되는 시드 고정

    # PyTorch가 사용할 수 있는 모든 CUDA 연산을 위한 결정론적 알고리즘을 사용
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_force_data():
    """
    force.dat 파일을 읽어서 마지막 줄의 시간, 항력 계수, 양력 계수를 반환합니다.
    """
    # 파일 경로 설정
    file_path = 'path_to_your_force.dat'

    # 파일 열기 및 읽기
    with open(file_path, 'r') as file:
        lines = file.readlines()  # 모든 줄을 읽어 리스트로 저장

    # 마지막 데이터 줄 추출
    last_line = lines[-1]  # 마지막 줄 가져오기
    data = last_line.split()  # 공백으로 분리

    # 필요한 데이터 추출
    time = float(data[0])  # 시간
    Cd = float(data[2])    # 항력 계수
    Cl = float(data[3])    # 양력 계수

    # 결과 출력
    print(f"Time: {time}, Drag Coefficient (Cd): {Cd}, Lift Coefficient (Cl): {Cl}")
