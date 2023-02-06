import numpy as np

def Analytic_ER_rule(r,w,p):
    m, n = np.shape(p)[0], np.shape(p)[1]
    P_fin = np.zeros(n)
    k1 = 0
    for i in range(n):
        temp = 1
        for j in range(m):
            temp=((w[j]*p[j][i]+1-r[j])/(1+w[j]-r[j]))*temp
        k1=k1+temp
    k2=1
    for i in range(m):
        k2=((1-r[i])/(1+w[i]-r[i]))*k2
    k2=(n-1)*k2
    k=k1-k2
    k=1/k

    for i in range(n):
        sec1=1
        sec2=1
        for j in range(m):
            sec1=((w[j]*p[j][i]+1-r[j])/(1+w[j]-r[j]))*sec1
            sec2=((1-r[j])/(1+w[j]-r[j]))*sec2
        numerator=k*(sec1-sec2)
        denominator=1-(k*sec2)
        P_fin[i]=numerator/denominator
    return P_fin
