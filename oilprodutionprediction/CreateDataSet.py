import numpy as np
import pandas as pd
from datetime import datetime
from scipy import interpolate
from keras.regularizers import l2  # 导入正则化l2（小写L）


class CreateDataSet():
    def __init__(self):
        self.upper = 0.3
        self.lower = 0.8

    def FilterData(self, data_frame, seqLength, predLength):
        '''
        过滤数据
        :param: dataFrame:
        :return:适合LSTM训练的数组
        '''
        mincount = seqLength + predLength
        # minCount= 20
        jh_arr = np.array(data_frame['JH']).astype(str)
        value_arr = np.array(data_frame.values[:, 2:])
        date_arr = np.array(data_frame['SJ']).astype(str)
        old_jh = jh_arr[0]
        out_array = []
        sub_item = []
        sub_date = []
        out_jh = []
        out_rq = []
        out_count = 0
        is_first = True
        number = 0
        for i in range(len(jh_arr)):
            current_jh = jh_arr[i]
            if current_jh != old_jh:
                if len(sub_item) > mincount:
                    input_item, jh_sub, rq_sub = self.__interpolationall(out_count, sub_item, sub_date, old_jh)
                    out_array.append(input_item)
                    out_jh.append(jh_sub)
                    out_rq.append(rq_sub)
                sub_item.clear()
                sub_date.clear()
                old_jh = current_jh
            list_value = value_arr[i]
            o_count = self.__single_np(list_value, 0)
            if list_value[-1] == 0:
                # o_count > 0 or
                number += 1
            else:
                out_date = datetime.strptime(date_arr[i], "%Y%m")#时间与井号目前是对应的
                sub_date.append(out_date)
                if is_first:
                    out_count = len(list_value)
                    is_first = False
                sub_item.append(list_value)
        if len(sub_item) > mincount:
            input_item, jh_sub, rq_sub = self.__interpolationall(out_count, sub_item, sub_date, old_jh)
            out_array.append(input_item)
            out_jh.append(jh_sub)
            out_rq.append(rq_sub)
        return out_array, out_count, out_jh, out_rq

    def __interpolationitem(self, x_arr, y_arr, x_pre):
        tck = interpolate.splrep(x_arr, y_arr, k=1)
        y_bspline = interpolate.splev(x_pre, tck)
        return y_bspline

    def __interpolationall(self, out_count, sub_item, sub_date, jh_str):
        sub_item = np.array(sub_item)
        sub_item = self.__smoothing_function(sub_item)  # 给数据加了一个平滑函数
        start_date = sub_date[0]
        end_date = sub_date[-1]
        all_month = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1
        x_pre = range(0, len(sub_item))
        jh_out = []
        rq_out = []
        for i in range(len(sub_item)):
            jh_out.append(jh_str)
            # q, r = divmod(start_date.month + i, 12)#divmod() 函数把除数和余数运算结果结合起来，返回一个包含商和余数的元组(a // b, a % b)。
            # if r == 0:
            #     q = q - 1
            #     r = 12
            rq_date = sub_date[i]
            rq_out.append(rq_date)

        x_old = []
        for item in range(len(sub_date)):
            # x_d = (item.year - start_date.year) * 12 + item.month - start_date.month
            x_d = item
            x_old.append(x_d)

        x_old = np.array(x_old)
        df_out = pd.DataFrame()
        for i in range(out_count):
            y_d_arr = sub_item[:, i]
            y_pre = self.__interpolationitem(x_old, y_d_arr, x_pre)
            df_out[str(i) + 'col'] = np.array(y_pre)

        return df_out.values, jh_out, rq_out

    def __smoothing_function(self, data):
        """
        :param data: 传入的单井数据
        :return: 平滑后的单井数据
        前后两个点设置阈值，超过1.2倍，就人为是离群点，改为平滑值
        """
        data = np.array(data)
        df = data[:, -1]
        for i in range(1, len(df) - 1, 1):
            z = 0.5 * (df[i - 1] + df[i + 1])
            if (z * (1 + self.upper)) < df[i] < (z * (1 - self.lower)):
                df[i] = z
        data[:, -1] = df
        out = data
        return out

    def __single_np(self, arr, target):
        # 计算列表或者array中某个元素出现的次数
        arr = np.array(arr)
        mask = (arr == target)
        arr_new = arr[mask]
        return arr_new.size