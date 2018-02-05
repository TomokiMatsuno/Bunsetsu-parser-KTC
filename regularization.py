import dynet as dy
import numpy as np
import config


class RegParams(object):
    def __init__(self, param):
        self._param = param
        # self._param_prev = np.array()
        self.set_prev_param()

    def set_prev_param(self):
        self._param_prev = dy.parameter(self._param).npvalue()

    def get_loss(self):
        ret = config.params_delta * dy.squared_distance(dy.parameter(self._param), dy.inputTensor(self._param_prev))
        self.set_prev_param()
        return ret