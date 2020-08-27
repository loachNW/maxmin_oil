import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.models import Sequential
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from matplotlib import pyplot as plt
from keras import initializers
from keras import optimizers

from keras.regularizers import l2  # 导入正则化l2（小写L）
import matplotlib as mpl
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


class BuildModel():
    def __init__(self):
        '''
        搭建模型
        '''
        self.modelID = ''
        # self.scaleD = StandardScaler()
        self.scaleD = MinMaxScaler(feature_range=(0, 1))
        self.inputCount = 0
        self.rate = 0.001
        self.SeqLength = 3
        self.PredLength = 1
        self.Batch = 16
        self.Epochs= 1

    def CreatTrainData(self, data_array, sample_jh, sample_rq):
        '''
        创建训练样本
        :param isRandom:
        :return:
        '''
        #改变样本数量
        #data_array = data_array[0:500]
        #sample_jh=sample_jh[0:500]
        #sample_rq=sample_rq[0:500]
        # 所有数据进行标准化处理
        out_x_train = []
        out_y_train = []
        out_jh_arr = []
        out_rq_arr = []
        if len(data_array) == 1:
            unit_array = self.scaleD.fit_transform(data_array[0])
            out_x_train, out_y_train, out_jh_arr, out_rq_arr = self.__makeSample(unit_array, sample_jh[0], sample_rq[0])
        else:
            all_data = []
            is_first = True
            for i in range(len(data_array)):
                if is_first:
                    all_data = data_array[i]
                    is_first = False
                else:
                    all_data = np.concatenate([all_data, data_array[i]])
            self.scaleD.fit_transform(all_data)
            del all_data

            is_first = True
            for i in range(len(data_array)):
                unit_array = self.scaleD.transform(data_array[i])
                sub_x_train, sub_y_train, sub_jh, sub_rq = self.__makeSample(unit_array, sample_jh[i], sample_rq[i])
                if is_first:
                    out_x_train = sub_x_train
                    out_y_train = sub_y_train
                    out_jh_arr = sub_jh
                    out_rq_arr = sub_rq
                    is_first = False
                else:
                    out_x_train = np.concatenate([out_x_train, sub_x_train])
                    out_y_train = np.concatenate([out_y_train, sub_y_train])
                    out_jh_arr = np.concatenate([out_jh_arr, sub_jh])
                    out_rq_arr = np.concatenate([out_rq_arr, sub_rq])

        # 样本索引混淆
        # index = [i for i in range(len(out_x_train))]
        # random.shuffle(index)
        # out_x_train = np.array(out_x_train[index])
        # out_y_train = np.array(out_y_train[index])
        # out_jh_arr = np.array(out_jh_arr[index])
        # out_rq_arr = np.array(out_rq_arr[index])
        return out_x_train, out_y_train, out_jh_arr, out_rq_arr

    def CreatTestData(self, data_array, sample_jh, sample_rq):
        '''
        创建训练样本
        :param isRandom:
        :return:
        '''
        # 所有数据进行标准化处理
        out_x_train = []
        out_y_train = []
        out_jh_arr = []
        out_rq_arr = []


        is_first = True
        for i in range(len(data_array)):
            unit_array = self.scaleD.transform(data_array[i])
            sub_x_train, sub_y_train, sub_jh, sub_rq = self.__makeSample(unit_array, sample_jh[i], sample_rq[i])
            if is_first:
               out_x_train = sub_x_train
               out_y_train = sub_y_train
               out_jh_arr = sub_jh
               out_rq_arr = sub_rq
               is_first = False
            else:
               out_x_train = np.concatenate([out_x_train, sub_x_train])
               out_y_train = np.concatenate([out_y_train, sub_y_train])
               out_jh_arr = np.concatenate([out_jh_arr, sub_jh])
               out_rq_arr = np.concatenate([out_rq_arr, sub_rq])

        # 样本索引混淆
        # index = [i for i in range(len(out_x_train))]
        # random.shuffle(index)
        # out_x_train = np.array(out_x_train[index])
        # out_y_train = np.array(out_y_train[index])
        # out_jh_arr = np.array(out_jh_arr[index])
        # out_rq_arr = np.array(out_rq_arr[index])
        return out_x_train, out_y_train, out_jh_arr, out_rq_arr

    def __makeSample(self, sub_data, sub_jh, sub_rq):
        '''
        按照每口井独自形成样本
        :param sub_data:
        :return:
        '''
        x_array = np.array(sub_data)
        y_array = np.array(sub_data)[:, -1]
        sequence_length = self.SeqLength + self.PredLength
        all_count = len(y_array)
        x_data = []
        y_data = []
        jh_out = []
        rq_out = []
        for i in range(0, all_count - sequence_length):
            x_sub = np.array(x_array[i:i + self.SeqLength, :])
            x_data.append(x_sub)
            y_data.append(y_array[i + self.SeqLength:i + sequence_length])
            jh_out.append(sub_jh[i])
            rq_out.append(sub_rq[i + self.SeqLength])

        out_x_train = np.array(x_data)
        out_y_train = np.array(y_data)
        jh_out= np.array(jh_out)
        rq_out= np.array(rq_out)
        return out_x_train, out_y_train, jh_out, rq_out


    # 建立模型
    def BuildModel(self, layers,):
        '''
        创建模型
        :param layers: 每个网络层的节点数。
        :return: 创建的训练模型对象
        '''
        inputCount = 5
        learning_rate = 0.001  # 学习率

        model = Sequential()
        # 第1层，LSTM，结点个数为特征个数；
        model.add(LSTM(input_shape=(self.SeqLength, self.inputCount), kernel_initializer=initializers.he_uniform(seed=7),  output_dim=layers[0], return_sequences=True))
        model.add(Activation('relu'))

        for i in range(1, len(layers) - 1):
            model.add(LSTM(layers[i], return_sequences=True,kernel_regularizer=l2(0.01)))#加入正则化
            model.add(Dropout(0.2))
        model.add(LSTM(layers[len(layers) - 1], return_sequences=False))
        model.add(Activation('relu'))
        # 倒数第2层，全连接，结点个数为128；
        #model.add(Dense(128, kernel_initializer=initializers.he_normal(seed=None)))
        # 倒数第1层，全连接，结点个数为预测长度；
        model.add(Dense(self.PredLength, kernel_initializer=initializers.he_uniform(seed=7)))
        #model.add(Dense(self.PredLength, kernel_initializer=initializers.he_normal(seed=7)))
        adam = optimizers.Adam(lr=self.rate)
        model.compile(loss='mse', optimizer=adam, metrics=['mae'])

        return model

    def BuildModel2(self, layers,init_mode='uniform'):
        '''
        创建模型
        :param layers: 每个网络层的节点数。
        :return: 创建的训练模型对象
        '''
        inputCount = 5
        learning_rate = 0.001  # 学习率
        model = Sequential()
        # 第1层，LSTM，结点个数为特征个数；
        model.add(LSTM(input_shape=(self.SeqLength, self.inputCount),kernel_initializer=init_mode, output_dim=layers[0], return_sequences=True))
        model.add(Activation('relu'))

        for i in range(1, len(layers) - 1):
            model.add(LSTM(layers[i], return_sequences=True,kernel_regularizer=l2(0.01)))
            model.add(Dropout(0.2))


        model.add(LSTM(layers[len(layers) - 1], return_sequences=False))
        model.add(Activation('relu'))
        # 倒数第2层，全连接，结点个数为128；
        #model.add(Dense(128, kernel_initializer=initializers.he_normal(seed=None)))
        # 倒数第1层，全连接，结点个数为预测长度；
        model.add(Dense(self.PredLength, kernel_initializer=initializers.he_normal(seed=7)))
        adam = optimizers.Adam(lr=self.rate)
        model.compile(loss='mse', optimizer=adam, metrics=['mae'])

        return model
    def my_loss(y_true, y_pred):
        loss1 = (y_true - y_pred)**2/len(y_true)
        return loss1

    def BuildModel1(self, layers,):
        '''
        创建模型，实现验证曲线
        :param layers: 每个网络层的节点数。
        :return: 创建的训练模型对象
        '''
        inputCount = 5
        learning_rate = 0.001  # 学习率
        model = Sequential()
        model.add(LSTM(input_shape=(self.SeqLength, inputCount), output_dim=64, return_sequences=True))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(LSTM(256, return_sequences=True))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(LSTM(256, return_sequences=False))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('tanh'))

        model.add(Dense(512, kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(512, kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(256, kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(256, kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(128, kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.PredLength))
        adam = optimizers.Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=adam, metrics=['mae'])
        return model


    def MyGridSearchCV(x_train, y_train):

        def BuildModel2(self ):
            '''
            创建模型，实现验证曲线
            :param layers: 每个网络层的节点数。
            :return: 创建的训练模型对象
            '''
            inputCount = 5
            learning_rate = 0.001  # 学习率
            model = Sequential()
            model.add(LSTM(input_shape=(self.SeqLength, inputCount), output_dim=64, return_sequences=True))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation('relu'))

            model.add(LSTM(256, return_sequences=True))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation('relu'))

            model.add(LSTM(256, return_sequences=False))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation('tanh'))

            model.add(Dense(512, kernel_regularizer=l2(0.01)))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation('relu'))
            model.add(Dropout(0.2))

            model.add(Dense(512, kernel_regularizer=l2(0.01)))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation('relu'))
            model.add(Dropout(0.2))

            model.add(Dense(256, kernel_regularizer=l2(0.01)))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation('relu'))
            model.add(Dropout(0.2))

            model.add(Dense(256, kernel_regularizer=l2(0.01)))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation('relu'))
            model.add(Dropout(0.2))

            model.add(Dense(128, kernel_regularizer=l2(0.01)))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation('relu'))
            model.add(Dropout(0.2))
            model.add(Dense(self.PredLength))
            adam = optimizers.Adam(lr=learning_rate)
            model.compile(loss='mse', optimizer=adam, metrics=['mae'])
            return model

        model = KerasClassifier(build_fn =BuildModel2(), verbose=0)
        batch_size = [10, 20]
        epochs = [10, 15]
        param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)  # 所有示例的配置为了实现并行化（n_jobs=-1）
        grid_result = grid.fit(x_train, y_train)
        # print(grid_result.grid_scores_, grid_result.best_params_, grid_result.best_score_)
        print(grid_result.best_estimator_)
        #pred = grid.predict(x_text)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


    def training_vis(self, hist):
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        # mae= hist.history['mae']
        # val_mae = hist.history['val_mae']
        # make a figure
        plt.rcParams['font.sans-serif'] = ['SimHei']
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来显示中文，防止乱码
        mpl.rcParams['font.family'] = 'sans-serif'  # 用来显示中文，防止乱码
        fig = plt.figure(figsize=(8, 4))

        # subplot loss
        ax1 = fig.add_subplot(121)
        ax1.plot(loss, label='训练集')
        ax1.plot(val_loss, label='验证集')
        ax1.set_xlabel('迭代次数', fontsize=12)
        ax1.set_ylabel('损失', fontsize=12)
        ax1.set_title('训练集和验证集损失变化曲线')
        ax1.legend()
        # subplot acc
        ax2 = fig.add_subplot(122)
        # ax2.plot(mae,label='训练集')
        # ax2.plot(val_mae,label='验证集')
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('平均绝对误差')
        ax2.set_title('训练集和验证集平均绝对误差变化曲线')
        ax2.legend()
        plt.tight_layout()
        plt.show()

    def FitModel(self, model, x_train, y_train, x_test, y_test):
        '''
        训练模型
        :param model: 训练模型对象
        :param x_train: 训练样本
        :param y_train: 样本对应的标签
        :return:
        '''
        save_best = ModelCheckpoint('D:\done\maxmin_oil\oilprodutionprediction\model_oil_.h5', verbose=1, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(patience=3, verbose=1, cooldown=1, factor=0.2)
        # early_stop = EarlyStopping(patience=5, verbose=1)
        # score = model.evaluate(x_test, y_test, verbose=1)
        # print(score)
        hist = model.fit(x_train, y_train, batch_size=self.Batch, nb_epoch=self.Epochs,validation_data=(x_test, y_test),callbacks=[reduce_lr, save_best])
        self.training_vis(hist)

    # 保存模型
    def SaveModel(self, model, modelpath):
        '''
        保存模型
        :param model: 模型对象
        :return:
        '''
        # 保存h5模型
        save_path = modelpath + '/' + str(self.modelID) + '.h5'
        model.save(save_path)

        # 保存配置参数
        std_arr = np.array(self.scaleD.scale_).astype(str)
        mean_arr = np.array(self.scaleD.mean_).astype(str)
        df_save = pd.DataFrame(columns=['std', 'mean'])
        std_str = ','.join(std_arr.tolist())
        mean_str = ','.join(mean_arr.tolist())
        df_save.loc[0] = [std_str, mean_str]
        save_path = modelpath + '/' + str(self.modelID) + '.csv'
        df_save.to_csv(save_path, index=False)