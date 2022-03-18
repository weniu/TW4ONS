import numpy as np
def MED(num,SDD,OriginPatternSave):
    """
    最小欧式距离
    :param num: 聚类中心数
    :param SDD: 待分类数据
    :param OriginPatternSave: K值
    :return: 分类的类型
    """
    Distance_WithKIII = np.zeros(num)
    for i in range(0,num):
        Distance_WithKIII[i] = np.sum(np.sum((SDD - OriginPatternSave[i]) ** 2))
    index = np.argmin(Distance_WithKIII)
    return index