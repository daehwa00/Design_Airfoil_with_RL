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

