import numpy as np

def RungeKutta4(f,y,y0,DELTA):
    TT=np.size(y,axis=1)
    for ii in np.arange(0,TT-1,DELTA):
        if ii==0:
            y[:,ii]=y0
        k1=DELTA*f(ii,y[:,ii])
        k2 = DELTA * f(ii + DELTA / 2, y[:, ii] + k1 / 2)
        k3 = DELTA * f(ii + DELTA / 2, y[:, ii] + k2 / 2)
        k4 = DELTA * f(ii + DELTA, y[:, ii] + k3)
        y[:, ii + 1] = y[:, ii] + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
    y=y.T
    return y

def Euler(f,y,y0,DELTA):
    TT=np.size(y,axis=1)
    y[:,0]=y0
    for ii in np.arange(1,TT,DELTA):
        y[:, ii] = y[:, ii - 1] + DELTA * f(ii - 1, y[:, ii - 1])
    y=y.T
    return y