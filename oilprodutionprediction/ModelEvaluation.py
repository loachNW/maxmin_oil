import numpy as np

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import scipy as sp
from sklearn.model_selection import validation_curve
from keras.regularizers import l2  # 导入正则化l2（小写L）


class ModelEvaluation:
    def __init__(self):
        self.real = []
        self.pre = []

    def __resultRelution(self):
        """
        :param real: 真实数据
        :param pre: 预测数据
        :return: 精确率
        公式： 精确率 = 1-[sum(abs(real-pre)/real)/len(real)]
        """
        real = np.array(self.real)
        pre = np.array(self.pre)

        subtraction = np.abs(real - pre)
        score = subtraction / real
        sum_score = np.sum(score)
        out1 = 1 - (sum_score / len(real))
        count = 0
        for i in score:
            if i < 0.1:
                count += 1
        out2 = count / len(score)
        return out1, out2

    def resultEstimate(self):
        out_dict = {}
        mae = mean_absolute_error(self.real, self.pre)
        mae = round(mae, 3)
        mae_list = []
        mae_list.append(mae)
        out_dict["MAE"] = mae_list

        #mape = mean_absolute_error(self.real, self.pre)
        mape = np.mean(np.abs((self.pre - self.real) / self.real)) * 100
        mape = round(mape, 2)
        mape_list = []
        mape_list.append(mape)
        out_dict["MAPE"] = mape_list

        r2 = r2_score(self.real, self.pre)
        r2 = round(r2, 3)
        r2_list = []
        r2_list.append(r2)
        out_dict["R2"] = r2_list

        rmse = sp.sqrt(sp.mean((self.real - self.pre) ** 2))
        rmse = round(rmse, 3)
        rmse_list = []
        rmse_list.append(rmse)
        out_dict["RMSE"] = rmse_list

        out1, out2 = self.__resultRelution()
        out_dict["准确率"] = out1
        out_dict["精确率"] = out2

        out_err = pd.DataFrame(out_dict)
        return out_err