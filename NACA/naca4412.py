import numpy as np


# NACA 0012 (160 points)
naca4412 = {
    "x": np.array(
        [
            1.0000,
            0.9500,
            0.9000,
            0.8000,
            0.7000,
            0.6000,
            0.5000,
            0.4000,
            0.3000,
            0.2500,
            0.2000,
            0.1500,
            0.1000,
            0.0750,
            0.0500,
            0.0250,
            0.0125,
            0.0000,
            0.0000,
            0.0125,
            0.0250,
            0.0500,
            0.0750,
            0.1000,
            0.1500,
            0.2000,
            0.2500,
            0.3000,
            0.4000,
            0.5000,
            0.6000,
            0.7000,
            0.8000,
            0.9000,
            0.9500,
            1.0000,
        ]
    ),
    "y": np.array(
        [
            0.0000,
            0.0147,
            0.0271,
            0.0489,
            0.0669,
            0.0814,
            0.0919,
            0.0980,
            0.0976,
            0.0941,
            0.0880,
            0.0789,
            0.0659,
            0.0576,
            0.0473,
            0.0339,
            0.0244,
            0.0000,
            0.0000,
            -0.0143,
            -0.0195,
            -0.0249,
            -0.0274,
            -0.0286,
            -0.0288,
            -0.0274,
            -0.0250,
            -0.0226,
            -0.0180,
            -0.0140,
            -0.0100,
            -0.0065,
            -0.0039,
            -0.0022,
            -0.0016,
            -0.0013,
        ]
    ),
}


data_4412 = {
    "RN": [4.2e4, 8.3e4, 1.6e5, 3.3e5, 6.4e5],
    "AOA": [
        -7.0,
        -6.0,
        -5.0,
        -4.0,
        -3.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        20.0,
        21.0,
    ],
    "CL": {
        4.2e4: [
            None,
            -0.27,
            -0.15,
            -0.03,
            0.08,
            0.19,
            0.30,
            0.40,
            0.50,
            0.59,
            0.67,
            0.74,
            0.80,
            0.86,
            0.91,
            0.96,
            0.99,
            1.02,
            1.03,
            1.03,
            1.02,
            1.00,
            0.96,
            0.93,
            0.91,
            0.90,
            0.89,
            0.88,
            None,
        ],
        8.3e4: [
            None,
            0.18,
            -0.10,
            0.00,
            0.10,
            0.21,
            0.32,
            0.42,
            0.53,
            0.63,
            0.72,
            0.81,
            0.89,
            0.95,
            1.02,
            1.07,
            1.12,
            1.16,
            1.18,
            1.20,
            1.19,
            1.17,
            1.14,
            1.10,
            1.06,
            1.03,
            1.00,
            0.98,
            0.96,
        ],
        1.6e5: [
            None,
            -0.23,
            -0.10,
            0.02,
            0.13,
            0.24,
            0.34,
            0.43,
            0.52,
            0.61,
            0.70,
            0.77,
            0.85,
            0.91,
            0.98,
            1.04,
            1.09,
            1.13,
            1.17,
            1.19,
            1.20,
            1.19,
            1.17,
            1.14,
            1.10,
            1.07,
            1.04,
            1.02,
            None,
        ],
        3.3e5: [
            -0.26,
            -0.19,
            -0.11,
            -0.03,
            0.07,
            0.17,
            0.27,
            0.37,
            0.47,
            0.56,
            0.65,
            0.74,
            0.82,
            0.90,
            0.98,
            1.05,
            1.11,
            1.17,
            1.22,
            1.26,
            1.29,
            1.29,
            1.25,
            1.18,
            1.12,
            1.08,
            1.05,
            1.03,
            None,
        ],
        6.4e5: [
            None,
            -0.21,
            -0.11,
            0.00,
            0.10,
            0.20,
            0.30,
            0.41,
            0.51,
            0.60,
            0.69,
            0.77,
            0.85,
            0.92,
            0.99,
            1.05,
            1.11,
            1.17,
            1.23,
            1.29,
            1.34,
            1.37,
            1.38,
            1.36,
            1.33,
            1.29,
            1.23,
            1.17,
            None,
        ],
    },
    "CD": {
        4.2e4: [
            None,
            0.0532,
            0.0431,
            0.0373,
            0.0353,
            0.0352,
            0.0345,
            0.0334,
            0.0325,
            0.0316,
            0.0308,
            0.0305,
            0.0307,
            0.0315,
            0.0331,
            0.0356,
            0.0387,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        8.3e4: [
            None,
            None,
            None,
            None,
            0.0327,
            0.0291,
            0.0251,
            0.0233,
            0.0229,
            0.0230,
            0.0236,
            0.0245,
            0.0259,
            0.0279,
            0.0308,
            0.0346,
            0.0398,
            0.0470,
            0.0568,
            0.0704,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        1.6e5: [
            None,
            0.0305,
            0.0223,
            0.0180,
            0.0159,
            0.0150,
            0.0147,
            0.0145,
            0.0144,
            0.0144,
            0.0144,
            0.0146,
            0.0151,
            0.0161,
            0.0176,
            0.0199,
            0.0235,
            0.0288,
            0.0370,
            0.0504,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        3.3e5: [
            None,
            0.0193,
            0.0166,
            0.0149,
            0.0140,
            0.0136,
            0.0131,
            0.0126,
            0.0122,
            0.0120,
            0.0120,
            0.0123,
            0.0130,
            0.0141,
            0.0157,
            0.0182,
            0.0219,
            0.0277,
            0.0373,
            0.0543,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        6.4e5: [
            None,
            0.0147,
            0.0135,
            0.0126,
            0.0120,
            0.0117,
            0.0117,
            0.0119,
            0.0123,
            0.0127,
            0.0131,
            0.0137,
            0.0144,
            0.0152,
            0.0163,
            0.0177,
            0.0198,
            0.0231,
            0.0284,
            0.0369,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
    },
}
