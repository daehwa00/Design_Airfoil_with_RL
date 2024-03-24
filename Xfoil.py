from xfoil import XFoil
from xfoil.test import naca0012

import matplotlib.pyplot as plt

xf = XFoil()
xf.airfoil = naca0012  # 예: NACA 0012 에어포일
Re = 1e6  # 레이놀즈 수
alpha = 5  # 공격각 (도)

# XFOIL을 사용하여 에어포일의 성능 계산
xf.Re = 1e6 # 레이놀즈 수 (Re = rho * V * c / mu = 관성력 / 점성력)
xf.max_iter = 40    # 최대 반복 횟수

# 받음각, 양력계수, 항력계수, 모멘트계수, 최소 압력계수
a, cl, cd, cm, cp = xf.aseq(-20, 20, 0.5)   # angle of attack을 -20도에서 20도까지 0.5도 간격으로 변화시키면서 계산

plt.plot(a, cl)
plt.show()

cl, cd, cm = xf.a(10)
a, cd, cm = xf.cl(1)
a, cl, cd, cm, cp = xf.cseq(-0.5, 0.5, 0.05)