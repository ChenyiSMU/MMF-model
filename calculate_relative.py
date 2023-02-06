import numpy as np
import pandas as pd

def calculating_reliability_new(P, t):
    m, n = np.shape(P)[0], np.shape(P)[1]
    R = np.zeros(m)
    for i in range(m):
        part1 = 1
        num = 0
        if P[i][0] >= t[i]:
            for j in range(m):
                if j != i:
                    part1 = P[j][1]*part1
                    if P[j][0] >= t[j]:
                        num = num+1
            R[i] = (num/(m-1))*(1-part1)
        if P[i][0] < t[i]:
            for j in range(m):
                if j != i:
                    part1 = P[j][0]*part1
                    if P[j][0] < t[j]:
                        num = num+1
        R[i] = (num/(m-1))*(1-part1)
    return R