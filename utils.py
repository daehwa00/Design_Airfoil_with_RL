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

def run_simulation():
    """
    OpenFOAM을 사용하여 시뮬레이션을 실행합니다.
    """
    move_block_mesh_dict()
    os.system('blockMesh')
    Cd, Cl = run_and_read_force_data()

    return Cd, Cl

def run_and_read_force_data():
    """
    force.dat 파일을 읽어서 마지막 줄의 시간, 항력 계수, 양력 계수를 반환합니다.
    """
    simulation_directory = '/home/daehwa/OpenFOAM/daehwa-11/run/airfoil'
    os.chdir(simulation_directory)
    os.system('simpleFoam')

    # 결과 파일 경로
    result_file_path = os.path.join(simulation_directory, 'postProcessing/forces/0/force.dat')

    # 결과 파일 처리
    with open(result_file_path, 'r') as file:
        lines = file.readlines()

    # 마지막 데이터 줄 추출
    last_line = lines[-1]
    data = last_line.split()

    # 필요한 데이터 추출
    time = float(data[0])  # 시간
    Cd = float(data[2])    # 항력 계수
    Cl = float(data[3])    # 양력 계수

    return Cd, Cl

import shutil
import os


def move_block_mesh_dict():
    """
    blockMeshDict 파일을 생성 위치에서 시뮬레이션 디렉토리로 이동합니다.
    """
    # 파일 경로 설정
    source_path = './blockMeshDict'  # 현재 디렉토리
    destination_directory = '/home/daehwa/OpenFOAM/daehwa-11/run/airfoil'

    # 대상 경로 생성 (blockMeshDict 파일이 저장될 위치)
    destination_path = os.path.join(destination_directory, 'constant/polyMesh/blockMeshDict')

    # 파일 이동 (복사 후 원본 삭제를 원한다면 shutil.copy() 사용 후 os.remove()로 원본 삭제)
    shutil.move(source_path, destination_path)

    print(f"blockMeshDict has been moved to {destination_path}")
