import numpy as np
def quantification(m1trajectory,beginrc,endrc,controlparm = 50):
    #量化函数，量化神经元兴奋度
    NLOB=np.shape(m1trajectory)[1]
    SD = np.zeros(NLOB)
    lamda1=(endrc-beginrc)/5
    for i in range(0,NLOB):
        burst1 = m1trajectory[beginrc - 1:int(beginrc + lamda1), i]
        burst2 = m1trajectory[int(beginrc + lamda1):int(beginrc + 2 * lamda1), i]
        burst3 = m1trajectory[int(beginrc + 2 * lamda1):int(beginrc + 3 * lamda1), i]
        burst4 = m1trajectory[int(beginrc + 3 * lamda1):int(beginrc + 4 * lamda1), i]
        burst5 = m1trajectory[int(beginrc + 4 * lamda1):int(beginrc + 5 * lamda1), i]
        SD[i] = np.mean(
            [np.std(burst1, ddof=1), np.std(burst2, ddof=1), np.std(burst3, ddof=1),
             np.std(burst4, ddof=1),
             np.std(burst5, ddof=1)])
        flagSD = np.array(np.where(SD > controlparm))
        SD[flagSD] = np.min(SD)
    return SD