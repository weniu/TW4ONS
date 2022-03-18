import numpy as np
def Hebb_Learning(SD,kmmL,K=0.4,Heb = 1.2,hab_c = 0.9995):
    TT=400
    #赫布学习规则
    # n_row=np.shape(kmmL)[0]
    # n_column=np.shape(kmmL)[1]
    SDm=np.mean(SD)
    for i in range(0,50):
        for j in range(0,50):
            if(SD[i]>(1+K)*SDm) & (SD[j]>(1+K)*SDm)& (i!=j):
                kmmL[i,j]=Heb*kmmL[i,j]
            elif i ==j:
                kmmL[i,j]=0
            else:
                kmmL[i,j]=hab_c**TT*kmmL[i,j]
    return kmmL