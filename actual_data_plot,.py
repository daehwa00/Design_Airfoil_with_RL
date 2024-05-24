import matplotlib.pyplot as plt


# 주어진 데이터를 사용하여 데이터프레임 생성
data_0012 = {
    'RN': [1.7e5, 3.3e5, 6.6e5, 1.3e6, 2.4e6],
    'AOA': [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
    'CL': {
        1.7e5: [-0.28, -0.17, -0.07, 0.03, 0.13, 0.23, 0.32, 0.41, 0.49, 0.56, 0.63, 0.70, 0.75, 0.79, 0.82, 0.82, 0.82, 0.81, 0.80, 0.79, 0.77, None],
        3.3e5: [-0.30, -0.19, -0.08, 0.02, 0.12, 0.22, 0.31, 0.39, 0.47, 0.55, 0.63, 0.70, 0.76, 0.81, 0.84, 0.85, 0.84, 0.83, 0.81, 0.79, 0.77, None],
        6.6e5: [-0.27, -0.18, -0.09, 0.01, 0.11, 0.21, 0.31, 0.40, 0.50, 0.60, 0.70, 0.80, 0.89, 0.98, 1.06, 1.13, 0.99, 0.94, 0.91, 0.90, None, None],
        1.3e6: [-0.30, -0.19, -0.09, 0.02, 0.12, 0.22, 0.33, 0.43, 0.53, 0.63, 0.73, 0.82, 0.92, 1.00, 1.09, 1.16, 1.23, 1.30, 1.36, 1.41, 1.41, None],
        2.4e6: [-0.27, -0.18, -0.08, 0.02, 0.12, 0.23, 0.33, 0.43, 0.53, 0.63, 0.73, 0.83, 0.93, 1.03, 1.13, 1.23, 1.33, 1.43, 1.51, 1.53, 1.45, 1.17]
    },
    'CD': {
        1.7e5: [0.0129, 0.0121, 0.0124, 0.0132, 0.0143, 0.0152, 0.0150, 0.0134, 0.0121, 0.0127, 0.0177, 0.0258, 0.0328, 0.0365, None, None, None, None, None, None, None, None],
        3.3e5: [0.0109, 0.0108, 0.0104, 0.0100, 0.0098, 0.0098, 0.0100, 0.0105, 0.0113, 0.0124, 0.0138, 0.0157, 0.0182, 0.0216, None, None, None, None, None, None, None, None],
        6.6e5: [0.0110, 0.0106, 0.0104, 0.0104, 0.0107, 0.0111, 0.0116, 0.0123, 0.0130, 0.0138, 0.0148, 0.0160, 0.0177, 0.0206, 0.0266, None, None, None, None, None, None, None],
        1.3e6: [0.0098, 0.0105, 0.0101, 0.0095, 0.0099, 0.0105, 0.0109, 0.0115, 0.0122, 0.0130, 0.0139, 0.0150, 0.0162, 0.0178, 0.0197, 0.0220, 0.0245, 0.0277, 0.333, None, None, None],
        2.4e6: [0.0096, 0.0092, 0.0096, 0.0102, 0.0103, 0.0104, 0.0107, 0.0110, 0.0114, 0.0119, 0.0125, 0.0133, 0.0143, 0.0158, 0.0177, 0.0204, 0.0241, 0.0295, 0.0403, None, None, None]
    }
}

data_4412 = {
    'RN': [4.2e4, 8.3e4, 1.6e5, 3.3e5, 6.4e5],
    'AOA': [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
    'CL': {
        4.2e4: [None, -0.27, -0.15, -0.03, 0.08, 0.19, 0.30, 0.40, 0.50, 0.59, 0.67, 0.74, 0.80, 0.86, 0.91, 0.96, 0.99, 1.02, 
                  1.03, 1.03, 1.02, 1.00, 0.96, 0.93, 0.91, 0.90, 0.89, 0.88, None],
        8.3e4: [None, 0.18, -0.10, 0.00, 0.10, 0.21, 0.32, 0.42, 0.53, 0.63, 0.72, 0.81, 0.89, 0.95, 1.02, 1.07, 1.12, 1.16, 
                  1.18, 1.20, 1.19, 1.17, 1.14, 1.10, 1.06, 1.03, 1.00, 0.98, 0.96],
        1.6e5: [None, -0.23, -0.10, 0.02, 0.13, 0.24, 0.34, 0.43, 0.52, 0.61, 0.70, 0.77, 0.85, 0.91, 0.98, 1.04, 1.09, 1.13, 
                  1.17, 1.19, 1.20, 1.19, 1.17, 1.14, 1.10, 1.07, 1.04, 1.02, None],
        3.3e5: [-0.26, -0.19, -0.11, -0.03, 0.07, 0.17, 0.27, 0.37, 0.47, 0.56, 0.65, 0.74, 0.82, 0.90, 0.98, 1.05, 1.11, 1.17, 
                  1.22, 1.26, 1.29, 1.29, 1.25, 1.18, 1.12, 1.08, 1.05, 1.03, None],
        6.4e5: [None, -0.21, -0.11, 0.00, 0.10, 0.20, 0.30, 0.41, 0.51, 0.60, 0.69, 0.77, 0.85, 0.92, 0.99, 1.05, 1.11, 1.17, 
                  1.23, 1.29, 1.34, 1.37, 1.38, 1.36, 1.33, 1.29, 1.23, 1.17, None]
    },
    'CD': {
        4.2e4: [None, 0.0532, 0.0431, 0.0373, 0.0353, 0.0352, 0.0345, 0.0334, 0.0325, 0.0316, 0.0308, 0.0305, 0.0307, 0.0315, 
                  0.0331, 0.0356, 0.0387, None, None, None, None, None, None, None, None, None, None, None, None],
        8.3e4: [None, None, None, None, 0.0327, 0.0291, 0.0251, 0.0233, 0.0229, 0.0230, 0.0236, 0.0245, 0.0259, 0.0279, 0.0308, 
                  0.0346, 0.0398, 0.0470, 0.0568, 0.0704, None, None, None, None, None, None, None, None, None],
        1.6e5: [None, 0.0305, 0.0223, 0.0180, 0.0159, 0.0150, 0.0147, 0.0145, 0.0144, 0.0144, 0.0144, 0.0146, 0.0151, 0.0161, 
                  0.0176, 0.0199, 0.0235, 0.0288, 0.0370, 0.0504, None, None, None, None, None, None, None, None, None],
        3.3e5: [None, 0.0193, 0.0166, 0.0149, 0.0140, 0.0136, 0.0131, 0.0126, 0.0122, 0.0120, 0.0120, 0.0123, 0.0130, 0.0141, 
                  0.0157, 0.0182, 0.0219, 0.0277, 0.0373, 0.0543, None, None, None, None, None, None, None, None, None],
        6.4e5: [None, 0.0147, 0.0135, 0.0126, 0.0120, 0.0117, 0.0117, 0.0119, 0.0123, 0.0127, 0.0131, 0.0137, 0.0144, 0.0152, 
                  0.0163, 0.0177, 0.0198, 0.0231, 0.0284, 0.0369, None, None, None, None, None, None, None, None, None]
    }
}

fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(15, 15))

# NACA 0012 Data
for rn in data_0012['RN']:
    ax1[0].plot(data_0012['AOA'][:len(data_0012['CL'][rn])], data_0012['CL'][rn], label=f'Re = {rn:.1e}')
ax1[0].set_xlabel('AOA (degrees)')
ax1[0].set_ylabel('CL')
ax1[0].set_title('NACA 0012 CL vs AOA')
ax1[0].legend()
ax1[0].grid(True)

for rn in data_0012['RN']:
    CL = data_0012['CL'][rn]
    CD = data_0012['CD'][rn]
    L_D_ratio = [cl / cd for cl, cd in zip(CL, CD) if cl is not None and cd is not None]
    ax1[1].plot(data_0012['AOA'][:len(L_D_ratio)], L_D_ratio, label=f'Re = {rn:.1e}')
ax1[1].set_xlabel('AOA (degrees)')
ax1[1].set_ylabel('CL/CD')
ax1[1].set_title('NACA 0012 CL/CD vs AOA')
ax1[1].legend()
ax1[1].grid(True)

# NACA 4412 Data
for rn in data_4412['RN']:
    ax2[0].plot(data_4412['AOA'][:len(data_4412['CL'][rn])], data_4412['CL'][rn], label=f'Re = {rn:.1e}')
ax2[0].set_xlabel('AOA (degrees)')
ax2[0].set_ylabel('CL')
ax2[0].set_title('NACA 4412 CL vs AOA')
ax2[0].legend()
ax2[0].grid(True)

for rn in data_4412['RN']:
    CL = data_4412['CL'][rn]
    CD = data_4412['CD'][rn]
    L_D_ratio = [cl / cd for cl, cd in zip(CL, CD) if cl is not None and cd is not None]
    ax2[1].plot(data_4412['AOA'][:len(L_D_ratio)], L_D_ratio, label=f'Re = {rn:.1e}')
ax2[1].set_xlabel('AOA (degrees)')
ax2[1].set_ylabel('CL/CD')
ax2[1].set_title('NACA 4412 CL/CD vs AOA')
ax2[1].legend()
ax2[1].grid(True)

plt.tight_layout()
plt.show()
