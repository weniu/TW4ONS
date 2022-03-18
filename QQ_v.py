import numpy as np
def QQ_v(x,qlr):
    """
    Q(xi(t),qj):Nonlinear S-type input\output function derived from H-H equation
    :param x: Current state of neurons
    :param qlr: the maximum asymptote of the sigmoid function
    :return:Q(xi(t),qj) calculation result
    """
    qval = qlr * (1 - np.exp(-(np.exp(x) - 1) / qlr))
    return qval