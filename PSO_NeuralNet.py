# -*- coding: utf-8 -*-

# 预测0，1版本,iris数据
#更改x，y即可
import numpy as np
# import pandas as pd
# from scipy.spatial.distance import pdist #距离度量
from numpy import *
from sklearn import preprocessing
import xlrd
import matplotlib.pyplot as plt
# from PSO import PSO_NeuralNetwork
#
#
# # read data
# def excel_to_matrix(path, num):
#     table = xlrd.open_workbook(path).sheets()[num]  # 获取第一个sheet表
#     row = table.nrows  # 行数
#     col = table.ncols  # 列数
#     datamatrix = np.zeros((row, col))  # 生成一个nrows行ncols列，且元素均为0的初始矩阵
#     for x in range(col):
#         cols = np.matrix(table.col_values(x))  # 把list转换为矩阵进行矩阵操作
#         datamatrix[:, x] = cols  # 按列把数据存进矩阵中
#         # 数据归一化
#     #     min_max_scaler = preprocessing.MinMaxScaler()
#     #     datamatrix = min_max_scaler.fit_transform(datamatrix)
#     return datamatrix
#
#
def activation_function(choice, x):  # activation function ,output vector or num, x is a matrices
    if choice == 1:  # sigmoid

        return 1.0 / (1 + exp(-x))
    else:
        return np.maximum(x, 0.0)  # relu
        # return abs(x)


def function_derivative(choice, x):
    derivative = np.zeros([1, x.shape[1]])
    if choice == 1:
        return np.multiply((1 / (1 + exp(-x))), (1 - 1 / (1 + exp(-x))))  # 此处用点乘 * 不行，要用multiply       sigmoid
    else:
        for i in range(x.shape[1]):
            if x[0, i] > 0:
                derivative[0, i] = 1
            else:
                derivative[0, i] = 0
        return derivative


class NeuralNetwork:
    def __init__(self, x, y, alpha):  # 输入矩阵行都是行向量属性，列是样本数
        self.input = x
        self.y = y
        self.hidden_neuron_num = 4
        self.choice = 2  # 1:sigmoid, 2:relu,第一层用choice，第二层choice2
        self.choice2 = 1
        self.weights1 = np.random.rand(self.input.shape[1],
                                       self.hidden_neuron_num)  # 1*4     #initialize layer1 weights,hidden_neutal == 4 ,是矩阵
        # self.weights1 = np.random.uniform(1,50,(1,4))
        self.weights2 = np.random.rand(self.hidden_neuron_num, self.y.shape[1])  # 4*1
        # self.weights2 = np.random.uniform(1,50,(4,1))
        self.bias1 = np.random.rand(1, self.hidden_neuron_num)
        # self.bias1 = np.array([[0.1]])
        self.bias2 = np.random.rand(1, self.y.shape[1])
        # self.bias2 = np.array([[0.1]])
        self.output = np.zeros(self.y.shape[1])  # initialize output
        self.learning_rate = alpha
        self.loss = 0

    def feedforward(self, current_x):
        self.layer1 = activation_function(self.choice, np.dot(current_x, self.weights1) + self.bias1)  # current_x:1*1,self.weights:1*4,hidden_neural == 4
        self.output = activation_function(self.choice2,
                                          np.dot(self.layer1, self.weights2) + self.bias2)

    def backpropagation(self, current_x, current_y):
        d_weights2 = np.dot(self.layer1.T, np.multiply(2 * (self.output - mat(current_y)),
                                                       function_derivative(self.choice2,
                                                                           self.output)))  # 这里的 *是点乘，适合输出为多维情况
        d_weights1 = np.dot(mat(current_x).T, np.multiply(
            np.dot(np.multiply(function_derivative(self.choice2, self.output), 2 * (self.output - mat(current_y))),
                   self.weights2.T), function_derivative(self.choice, self.layer1)))
        d_bias2 = np.multiply(2 * (self.output - mat(current_y)), function_derivative(self.choice2, self.output))
        d_bias1 = np.multiply(
            np.dot(np.multiply(function_derivative(self.choice2, self.output), 2 * (self.output - mat(current_y))),
                   self.weights2.T), function_derivative(self.choice, self.layer1))

        # update weights and bias
        self.weights1 -= self.learning_rate * d_weights1
        self.weights2 -= self.learning_rate * d_weights2
        self.bias1 -= self.learning_rate * d_bias1
        self.bias2 -= self.learning_rate * d_bias2

    # def train(self):
    #     #         batch = 30
    #     #         i = random.randint(0,120)
    #     for i in range(self.input.shape[0]):
    #         self.feedforward(self.input[i])
    #         self.backpropagation(self.input[i], self.y[i])
    #         #         for i in range(self.input.shape[0]):
    #         #             self.feedforward(self.input[i])
    #         self.loss += np.sum(np.square(self.output - self.y[i])) / self.input.shape[0]

    def train(self):

        for i in range(self.input.shape[0]):
            self.feedforward(self.input[i])
            self.backpropagation(self.input[i], self.y[i])
        for j in range(self.input.shape[0]):
            self.feedforward(self.input[j])
            self.loss += np.sum(np.square(self.output - self.y[j])) / self.input.shape[0]

    def test(self, test_x, test_y):
        prediction = np.zeros((test_y.shape[0], test_y.shape[1]))
        for i in range(test_x.shape[0]):
            self.feedforward(test_x[i])
            prediction[i] = self.output[0]

        # fig = plt.figure()
        # aix = fig.add_subplot(1, 1, 1)
        # aix.plot(test_x, prediction, 'b-')
        #
        # plt.ioff()
        # plt.show()
        print(prediction)


# if __name__ == "__main__":
#     # x = np.linspace(-1,1,num = 300)[:,np.newaxis]
#     # noise = np.random.normal(0, 0.05, x.shape).astype(np.float32)
#     # y = np.square(x) - 0.5 + noise
#
#     x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
#     y = np.array([[0],[1],[1],[0]])
#
#     x_test = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0, 1, 1],[[1, 1, 0]]])
#     y_test = np.array([[0],[1],[1],[0],[1],[0]])
#     # x = np.array([[0,0],[1,0],[1,1],[0,1]])
#     # y = np.array([[0],[0],[1],[0]])
#
#     # datafile = "C:/Users/MaxMai/Desktop/python/iris_test+train.xlsx"
#
#     # x = excel_to_matrix(datafile, 0)
#     # y = excel_to_matrix(datafile, 3)
#
#     # x_test = excel_to_matrix(datafile, 1)
#     # y_test = excel_to_matrix(datafile, 2)
#
#     NN = PSO_NeuralNetwork(x, y)
#     NN.PSO_Optimizer()
#
#
    # for j in range(1500):
    #     NN.loss = 0
    #     NN.train()
    #
    #     if j % 50 == 0:
    #         print(NN.loss)
#
#
#     # fig = plt.figure()
#     # ax = fig.add_subplot(1, 1, 1)
#     # ax.scatter(x, y)
#     # plt.ion()  # 本次运行请注释，全局运行不要注释
#     # plt.show()
#
# #     NN.test(array([[0,0,1]]),array([[0]]))
# #     NN.test(array([[1, 0, 1]]), array([[1]]))
# #     NN.test(array([[1, 1, 1]]), array([[0]]))
# #     NN.test(array([[0, 0, 0]]), array([[0]]))
# #     NN.test(array([[0, 1, 1]]), array([[1]]))   #predict
# #     NN.test(array([[1, 1, 0]]), array([[0]]))   #predict
# # NN.test(array([[0, 1]]), array([[0]]))
# # NN.test(array([[1, 1]]), array([[1]]))
# #     NN.test(array([[6,2.9, 4.5, 1.5]]), array([[0,1,0]]))
#
#
# # print(loss)                             ##最好的误差
#     NN.test(x_test, y_test)