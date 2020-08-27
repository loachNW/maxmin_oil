import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from twoprodutionpredict.ModelEvaluation import ModelEvaluation
from twoprodutionpredict.twoBuildModel import BuildModel
from twoprodutionpredict.twoCreateDataSet import CreateDataSet
from keras.models import load_model

# 训练模型存储地址 str
SeqLength = 4
# 训练样本的输入维度
PredLength = 2
# 训练样本的输出维度
Epochs = 60
# 循环训练的轮次
Batch = 32
# 分块训练时，每块样本个数
lerrning_rate = 0.001
# 第1层，LSTM，结点个数为特征个数；
layerData = [64,64]
# 倒数第2层，全连接，结点个数为128；
# 倒数第1层，全连接，结点个数为预测长度；




def rm_main(data_frame1,data_frame2,):
    '''
    主函数
    :param data_frame: 样本数据
    :return:
    '''
    # 模型ID
    model_id = "chanliangyuce"
    # 数据的分割比例
    sample_spilt_scale = 0.9
    # 过滤掉因数据不足无法生成样本的井数据
    creatdata = CreateDataSet()

    wave_filtering = True
    diff = True
    if wave_filtering:
        data_frame1 = lvbo(data_frame1, "YCY", 5)
        data_frame2 = lvbo(data_frame2, "YCY", 5)
    if diff:
        data_frame1 = one_well_diff(data_frame1, False)
        data_frame2 = one_well_diff(data_frame2, False)#做不做差分影响不大
    train_out_value_arr, train_out_count, train_out_jh, train_out_rq = creatdata.FilterData(data_frame1, SeqLength, PredLength)
    test_out_value_arr, test_out_count, test_out_jh, test_out_rq = creatdata.FilterData(data_frame2, SeqLength, PredLength)
    #out_value_arr, out_count, out_jh, out_rq = creatdata.FilterData(data_frame, SeqLength, PredLength)
    # outValueArr插值后的连续特征 和 数据；outCount：样本的特征长度；outJH，outRQ：井号，日期。
    if len(train_out_value_arr) < 1:
        df_out = pd.DataFrame()
        df_out["Error"] = ["有效数据不足"]
        return df_out

    # 创建LSTM训练模型操作对象
    model_act = BuildModel()
    model_act.modelID = model_id
    model_act.inputCount = train_out_count
    # 每个样本的长度
    model_act.rate = lerrning_rate
    model_act.Batch = Batch
    model_act.SeqLength = SeqLength
    model_act.PredLength = PredLength
    model_act.Epochs = Epochs
    model = model_act.BuildModel(layerData)

    # 生成训练样本
    x_train, y_train, jh_train, rq_train = model_act.CreatTrainData(train_out_value_arr, train_out_jh, train_out_rq)

    # 生成测试样本
    x_test, y_testz, jh_test, rq_test = model_act.CreatTestData(test_out_value_arr, test_out_jh, test_out_rq)
    '''
    # 生成样本
    x_sample, y_sample, jh_sample, rq_sample = model_act.CreatTrainData(out_value_arr, out_jh, out_rq)
    # 分割样本
    split_count, x_train, y_train, x_test, y_test, jh_test, rq_test, zrlc_test = split_sample(x_sample,
                                                                                              y_sample,
                                                                                              jh_sample,
                                                                                              rq_sample,
                                                                                              sample_spilt_scale
                                                                                            )'''

    # 训练模型
    model_act.FitModel(model, x_train, y_train,x_test,y_testz)
    # 保存模型文件
    # model_act.SaveModel(model, modelpath)
    # 预测结果
    MODEL = load_model('D:/done/maxmin_oil/twoprodutionpredict/model_oil_.h5')
    y_predict = MODEL.predict(x_test)
    a = int(1)#0是含水率，1是日产油
    for zhiiao in range(a):
        y_predict_unit = y_predict[:, zhiiao]
        y_test = y_testz[:, zhiiao]
    # 预测结果保存
    y_pre = []
    y_te = []
    for i in range(len(y_predict_unit)):
        y_pre.append(y_predict_unit[i])
        y_te.append(y_test[i])


    x_max = model_act.scaleD.data_max_[-1]
    x_min = model_act.scaleD.data_min_[-1]
    scale = model_act.scaleD.data_min_[-1]
    y_pre = np.array(y_pre)*(x_max-x_min) + x_min
    y_te = np.array(y_te)*(x_max-x_min) + x_min

    # mean_y = model_act.scaleD.mean_[-1]
    # std_y = model_act.scaleD.scale_[-1]
    # y_pre = np.array(y_pre) * std_y + mean_y
    # y_te = np.array(y_te) * std_y + mean_y
    # 模型评估；‘
    df_out = pd.DataFrame()
    if len(x_test) > 1:
        df_out['jh'] = np.array(jh_test)
        df_out['rq'] = np.array(rq_test)
        df_out['real'] = np.array(y_te)
        df_out['predict'] = np.array(y_pre)
        df_out['jdwc'] = np.abs(np.array(y_pre) - np.array(y_te))
        df_out['xdwc'] = np.abs(np.array(y_pre) - np.array(y_te)) / np.array(y_te)
        df_out['mpe'] = (np.array(y_pre) - np.array(y_te)) / np.array(y_te)*100
        # dfOut = dfOut[(dfOut['real'] > 10)&(dfOut['predict'] > 10)]

    pingjia = ModelEvaluation()
    pingjia.pre = df_out['predict']
    pingjia.real = df_out['real']
    estimate_result = pingjia.resultEstimate()
    return df_out, estimate_result

#中值滤波
def lvbo(df,theme,k):
    win_len = k
    name_list = list(set(df["JH"].values.tolist()))
    output = pd.DataFrame(columns=df.columns.values)
    for name in name_list:
        ddf = df[df["JH"] == name]
        oil_y = ddf[theme].values.tolist()
        if len(oil_y) > win_len:
            pad_width = int((win_len - 1) / 2)
            out_y = medfiter(oil_y, pad_width)
            out_y = np.array(out_y).reshape(len(out_y),1)
            ddf[theme] = out_y
            output = output.append(ddf)
    return output

def medfiter(input_x, pad_width):
    xx = input_x[:pad_width]
    xx.extend(input_x)
    xx.extend(input_x[(-1 * pad_width):])
    out_y = []
    for i in range(pad_width, len(xx) - pad_width):
        out_y.append(np.median(xx[i - pad_width:i + pad_width + 1]))#
    return out_y

#差分
def one_well_diff(data,delete):
    well_name_list = data["JH"].values.tolist()
    well_names = set(well_name_list)
    out_data = pd.DataFrame(columns=data.columns.values)
    for well_name in well_names:
        child_data = data[data["JH"] == well_name]#找出井号对应的数据
        child_data_diff = __diffierenced(child_data,delete)
        out_data = out_data.append(child_data_diff)
    return out_data

def __diffierenced(df,delete):
    if delete:
        df_names = df.columns.values.tolist()
        out_pd = pd.DataFrame()
        out_pd[df_names[0]] = df[df_names[0]][1:]
        out_pd[df_names[1]] = df[df_names[1]][1:]
        for name in df_names[2:-2]:
            out_pd[name] = np.diff(df[name])
        out_pd[df_names[-2]] = df[df_names[-2]][1:]
        out_pd[df_names[-1]] = df[df_names[-1]][1:]
    else:
        df_names = df.columns.values.tolist()
        out_pd = pd.DataFrame()
        out_pd[df_names[0]] = df[df_names[0]][1:]#为什么不要第一个
        out_pd[df_names[1]] = df[df_names[1]][1:]
        for name in df_names[2:-1]:
            out_pd[name] = np.diff(df[name])
        out_pd[df_names[-1]] = df[df_names[-1]][1:]
    return out_pd

def split_sample(x_sample, y_sample, jh_sample, rq_sample, sample_spilt_scale):
    '''
    划分训练集和测试集
    :param x_sample: 输入样本
    :param y_sample: 输出样本
    :param jh_sample: 井号样本
    :param rq_sample: 日期样本
    :param sample_spilt_scale：训练集和验证集劈分比例
    :return split_count：训练集样本数
    :return x_train：训练集输入
    :return y_train：训练集输出
    :return x_test：测试集输入
    :return y_test：测试集输出
    :return jh_test：测试集井号
    :return rq_test：测试集日期
    :return zrlc_test：
    '''
    split_count = int(len(y_sample) * sample_spilt_scale)
    x_train = x_sample[:split_count]
    y_train = y_sample[:split_count]
    x_test = x_sample[split_count:]
    y_test = y_sample[split_count:]
    jh_test = jh_sample[split_count:]
    rq_test = rq_sample[split_count:]
    zrlc_test = np.zeros(len(rq_test))
    return split_count, x_train, y_train, x_test, y_test, jh_test, rq_test, zrlc_test

if __name__ == "__main__":

    #data_csv = "小孙最优参数-高含水.csv"
    #data_csv = "特高3个区块.csv"
    data_csv = "高含水样本2.csv"
    data_train_csv = "high-train.csv"
    data_test_csv = "high-test.csv"
    #data_train_csv = "高含水样本2 - train.csv"
    #data_test_csv = "高含水样本2 - test.csv"
    modelpath = 'D:/done/maxmin_oil/twoprodutionpredict/'

    path_train = modelpath + "/" + data_train_csv
    path_test = modelpath + "/" + data_test_csv
    #inputData = pd.read_csv(path_train_data, encoding="gbk", engine='python')
    data_train = pd.read_csv(path_train, encoding="gbk", engine='python')
    data_test = pd.read_csv(path_test, encoding="gbk", engine='python')
    layerData = [64, 64]
    pre_result, estimate = rm_main(data_train,data_test)
    path_train_result = modelpath + "/" +"预测结果.csv"
    pre_result.to_csv(path_train_result, encoding="utf_8_sig", index=False)
    path_estimate =modelpath + "/" + "误差比较.csv"
    estimate.to_csv(path_estimate, encoding="utf_8_sig", index=False)
    print(estimate)
    '''
    2、测试网络宽度影响
    layerData = [32, 32]
    pre_result, estimate = rm_main(data_frame=inputData)
    path_estimate = modelpath + "/" + "32.csv"
    estimate.to_csv(path_estimate, encoding="utf_8_sig", index=False)

    layerData = [16, 16]
    pre_result, estimate = rm_main(data_frame=inputData)
    path_estimate = modelpath + "/" + "16.csv"
    estimate.to_csv(path_estimate, encoding="utf_8_sig", index=False)

    layerData = [128, 128]
    pre_result, estimate = rm_main(data_frame=inputData)
    path_estimate = modelpath + "/" + "128.csv"
    estimate.to_csv(path_estimate, encoding="utf_8_sig", index=False)

    layerData = [256, 256]
    pre_result, estimate = rm_main(data_frame=inputData)
    path_estimate = modelpath + "/" + "256.csv"
    estimate.to_csv(path_estimate, encoding="utf_8_sig", index=False)

    layerData = [512, 512]
    pre_result, estimate = rm_main(data_frame=inputData)
    path_estimate = modelpath + "/" + "512.csv"
    estimate.to_csv(path_estimate, encoding="utf_8_sig", index=False)'''

    '''
    1、测试网络深度影响
    layerData = [64, 64, 64]
    pre_result, estimate = rm_main(data_train, data_test)
    path_estimate = modelpath + "/" + "3层.csv"
    estimate.to_csv(path_estimate, encoding="utf_8_sig", index=False)

    layerData = [64, 64, 64, 64]
    pre_result, estimate = rm_main(data_train, data_test)
    path_estimate = modelpath + "/" + "4层.csv"
    estimate.to_csv(path_estimate, encoding="utf_8_sig", index=False)

    layerData = [64, 64, 64, 64, 64]
    pre_result, estimate = rm_main(data_train, data_test)
    path_estimate = modelpath + "/" + "5层.csv"
    estimate.to_csv(path_estimate, encoding="utf_8_sig", index=False)

    layerData = [64, 64, 64, 64, 64, 64]
    pre_result, estimate = rm_main(data_train, data_test)
    path_estimate = modelpath + "/" + "6层.csv"
    estimate.to_csv(path_estimate, encoding="utf_8_sig", index=False)

    layerData = [64, 64, 64, 64, 64, 64, 64]
    pre_result, estimate = rm_main(data_train,data_test)
    path_estimate = modelpath + "/" + "7层.csv"
    estimate.to_csv(path_estimate, encoding="utf_8_sig", index=False)

    layerData = [64, 64, 64, 64, 64, 64, 64, 64]
    pre_result, estimate = rm_main(data_train,data_test)
    path_estimate = modelpath + "/" + "8层.csv"
    estimate.to_csv(path_estimate, encoding="utf_8_sig", index=False)

    layerData = [64, 64, 64, 64, 64, 64, 64, 64, 64]
    pre_result, estimate = rm_main(data_train, data_test)
    path_estimate = modelpath + "/" + "9层.csv"
    estimate.to_csv(path_estimate, encoding="utf_8_sig", index=False)

    layerData = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    pre_result, estimate = rm_main(data_train, data_test)
    path_estimate = modelpath + "/" + "10层.csv"
    estimate.to_csv(path_estimate, encoding="utf_8_sig", index=False)'''
