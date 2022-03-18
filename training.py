import numpy as np
import scipy.io as sio
from Model.KIII import KIII
import ODE
import LearningRule
import Quantification
import time
import Classification

def load_data(file='dct68_50.mat'):
    # 获取数据集
    dct8 = np.array(sio.loadmat(file)['dct8'].ravel()).reshape(5, -1)
    return dct8


def training(data_set,KIII, Hebb_Learning, quantification):
    start = time.time()
    """
    模型训练
    :param KIII: KIII模型
    :param Learning: 学习规则
    :param params: 需要训练的参数
    :param quantification: 量化函数
    :return:
    """
    # 获得数据集
    # data_set = load_data('dct81_50.mat')
    endTimes = 1
    # 设置循环次数，即数据大小
    endPatternSeq = 5  # 5个人的数据，5中模式


    # 获取数据集维度
    shape0,shape1 = data_set.shape
    shape2,shape3= data_set[0, 0].shape

    # endTrainingSample = shape1*0.6  # 训练数据的数量
    endTrainingSample = 1

    '''##############'''
    # OriginPatternRecord = np.zeros((shape0, shape1, shape2, shape3))
    OriginPattern = np.zeros((shape2, shape3))
    OriginPatternSave = np.zeros((shape0, shape2, shape3))

    for learnTimes in range(0, endTimes):
        for PatternSeq in range(0, endPatternSeq):  # Sequence Number of Recognized Pattern
            for trainingSample in range(0, endTrainingSample):
                for h in range(0, shape3):
                    pattern = data_set[PatternSeq, trainingSample][:, h]

                    y = KIII.forward(pattern, ODE.Euler)
                    m1trajectory = y[0:KIII.TT, 4 * KIII.NLOB:5 * KIII.NLOB]
                    SD = quantification(m1trajectory, beginrc=100, endrc=300)
                    # OriginPatternRecord[PatternSeq, trainingSample][:, h] = SD

                    # kmmL0 = KIII.kii_OB.kmmL
                    # '''##############'''
                    kmmL = Hebb_Learning(SD, KIII.kii_OB.kmmL)
                    KIII.kii_OB.kmmL = kmmL


                    if trainingSample == 0:
                        OriginPattern[:, h] = SD
                    else:
                        OriginPattern[:, h] = OriginPattern[:, h] + SD
                    OriginPatternSave[PatternSeq][:, h] = OriginPattern[:, h] / endTrainingSample

    end = time.time()
    print('Training_kiii Running time: %s Seconds' % (end - start))

    cognition_kiii(data_set,KIII, quantification, OriginPatternSave,Classification.MED)


def cognition_kiii(data_set,KIII, quantification, OriginPatternSave,classification):
    start = time.time()
    # 获得数据集
    # data_set = load_data('dct81_50.mat')

    # 获取数据集维度
    shape0, shape1 = data_set.shape
    shape2, shape3 = data_set[0, 0].shape

    endFolder = 5
    # startTestNumber = shape1*0.6
    startTestNumber = shape1-1
    endTestNumber = shape1

    #初始化定义
    # SD_record = np.zeros((shape0, shape1, shape2, shape3))
    SDD = np.zeros((shape2, shape3))
    # Distance_WithKIII = np.zeros(shape0)
    cogResult = np.zeros((shape0, shape1))

    for TestFolder in range(0, endFolder):
        for TestNumber in range(startTestNumber, endTestNumber):  # Sequence Number of Recognized Pattern
            for h in range(0, shape3):
                pattern = data_set[TestFolder, TestNumber][:, h]
                y = KIII.forward(pattern, ODE.Euler)

                m1trajectory = y[0:KIII.TT, 4 * KIII.NLOB:5 * KIII.NLOB]
                SD = quantification(m1trajectory, beginrc=100, endrc=300)

                # SD_record[TestFolder, TestNumber][:, h] = SD
                SDD[:, h] = SD
            #  识别的方式有多种：这里使用最小距离的方法进行判断。那个模式的距离最小的就认为是该模式，这样有些不是很科学。
            #  还有一种方法，即：如果小于某个距离才认为是该模式，否则就识别为无法判断，比如：找出最小距离何次小距离之差的多少分之一为门限。
            #  temp = energy_weight
            # {TestFolder, TestNumber};
            #  SD = SD. * temp;
            #  SDm = mean(SD);
            # for i in range(0, 5):
            #     Distance_WithKIII[i] = np.sum(np.sum((SDD - OriginPatternSave[i]) ** 2))
            # # [Mindistance, index] = np.min(Distance_WithKIII)
            # # Mindistance = np.min(Distance_WithKIII)
            # index = np.argmin(Distance_WithKIII)

            cogResult[TestFolder, TestNumber] = classification(endFolder,SDD,OriginPatternSave)

    # datestr(now)
    errorNum = 0
    for i in range(1, 5):
        for j in range(startTestNumber, endTestNumber):
            if cogResult[i, j] != i:
                errorNum = errorNum + 1

    errorRatio = errorNum / ((endTestNumber - startTestNumber) * 5)
    print("errorRatio: ", errorRatio)
    end = time.time()
    print('Cognitiong_kiii Running time: %s Seconds' % (end - start))


# def


kiii = KIII(NLOB=25)
data_set = load_data('dct68_25.mat')
training(data_set,kiii, LearningRule.Hebb_Learning, Quantification.quantification)
