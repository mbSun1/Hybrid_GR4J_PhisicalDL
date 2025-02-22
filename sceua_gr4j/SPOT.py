
import numpy as np
import spotpy
from sceua_gr4j.gr4j  import GR4J

class gr4j_spot_setup(object):
    def __init__(self, prec, etp,qobs):
        self.params = [
            spotpy.parameter.Uniform("x1", low=100, high=1200),
            spotpy.parameter.Uniform("x2", low=-3, high=5),
            spotpy.parameter.Uniform("x3", low=20, high=300),
            spotpy.parameter.Uniform("x4", low=1, high=3),
        ]

        # 降水、潜在蒸散发、实测径流数据
        self.prec = prec
        self.etp = etp
        self.qobs = qobs

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self,x):
        model = GR4J()
        # x1, x2, x3, x4 = model.init_params()
        x1, x2, x3, x4 = x[0],x[1], x[2], x[3]

        model.update_params(x1, x2, x3, x4)

        qsim,_, _= model.run(self.prec, self.etp)
        return qsim, _, _

    def evaluation(self):
        return self.qobs

    def objectivefunction(self, simulation, evaluation):
        qsim, _, _ = simulation
        qobs = evaluation

        # 转换为 NumPy 数组
        qsim = np.array(qsim)
        qobs = qobs.detach().numpy()
        # nse = 1 - np.sum((qobs - qsim) ** 2) / np.sum((qobs - np.mean(qobs)) ** 2)
        # nse = -1 + np.sum((qobs - qsim) ** 2) / np.sum((qobs - np.mean(qobs)) ** 2)
        # 计算 RMSE
        rmse = np.sqrt(np.mean((qobs - qsim) ** 2))


        # 计算 MSE
        # mse = np.mean((qobs - qsim) ** 2)

        return rmse




