import numpy as np

from numpy import *
from sklearn import preprocessing
import xlrd
import matplotlib.pyplot as plt
from PSO_NeuralNet import NeuralNetwork


# read data
def excel_to_matrix(path, num):
    table = xlrd.open_workbook(path).sheets()[num]  # 获取第一个sheet表
    row = table.nrows  # 行数
    col = table.ncols  # 列数
    datamatrix = np.zeros((row, col))  # 生成一个nrows行ncols列，且元素均为0的初始矩阵
    for x in range(col):
        cols = np.matrix(table.col_values(x))  # 把list转换为矩阵进行矩阵操作
        datamatrix[:, x] = cols  # 按列把数据存进矩阵中
        # 数据归一化
        min_max_scaler = preprocessing.MinMaxScaler()
        datamatrix = min_max_scaler.fit_transform(datamatrix)
    return datamatrix


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

class PSO_NeuralNetwork(NeuralNetwork):
    def __init__(self,x,y):

        NeuralNetwork.__init__(self,x,y,alpha=0.01)
        self.w = 0.7298
        self.c_1 = 1.5
        self.c_2 =1.5

        self.dim = self.weights1.shape[0]*self.weights1.shape[1] + self.weights2.shape[0]*self.weights2.shape[1] + self.bias1.shape[1] + self.bias2.shape[1]
        self.popsize = 25
        self.X_pop = np.zeros([self.popsize,self.dim])
        self.decpop = np.zeros([self.popsize,1])
        self.variable = np.zeros([1, self.dim + 1])
        self.V = np.zeros([self.popsize,self.dim])
        self.best_mat = np.zeros([self.popsize, self.dim + 1])
        self.cost = 1e4
        self.T = round(self.cost/self.popsize)
        self.best_y = np.zeros([1, self.T])
        self.upper_bound = 30
        self.lower_bound = -30
    #
    # def make_bounds(self):
    def feed_forward_pso(self, current_x):
        layer1 = activation_function(self.choice, np.dot(current_x, self.weights1) + self.bias1)  # current_x:1*1,self.weights:1*4,hidden_neural == 4
        output = activation_function(self.choice2,
                                          np.dot(layer1, self.weights2) + self.bias2)
        return output
    def object_function(self,entity):
        self.loss = 0
        count = 0
        for m in range(self.weights1.shape[0]):
            for n in range(self.weights1.shape[1]):
                self.weights1[m][n] = entity[count]
                count += 1
        for o in range(self.bias1.shape[1]):
            self.bias1[0][o] = entity[count]
            count += 1
        for p in range(self.weights2.shape[0]):
            for q in range(self.weights2.shape[1]):
                self.weights2[p][q] = entity[count]
                count +=1
        for r in range(self.bias2.shape[1]):
            self.bias2[0][r] = entity[count]
            count += 1
        for i in range(self.input.shape[0]):            #计算均方误差函数
            output = self.feed_forward_pso(self.input[i])
            self.loss += np.sum(np.square(output - self.y[i]))/self.input.shape[0]

        return  -self.loss


    def update_pop(self):
        g_best = self.calculate_best()  #完全图
        for i in range(self.popsize):
            p_best = self.best_mat[i][1:self.dim+1]
            self.V[i,:] = self.w *self.V[i,:] + self.c_1*random.rand()*(p_best - self.X_pop[i,:])+ self.c_2*random.rand()*(g_best - self.X_pop[i,:])
            self.X_pop[i,:] = self.X_pop[i,:] + self.V[i,:]

            for j in range(self.dim):                   #设置有无边界
                if self.X_pop[i][j] > self.upper_bound:
                    self.X_pop[i][j] = self.upper_bound
                elif self.X_pop[i][j] < self.lower_bound:
                    self.X_pop[i][j] = self.lower_bound


    def calculate_best(self):
        opt = max(self.best_mat[:,0])
        r = np.argwhere(self.best_mat[:,0] == opt)
        best_of_all = self.best_mat[r[0][0],1:self.dim+ 1]
        return best_of_all

    def PSO_Optimizer(self):
        for generation in range(self.T):
            if generation == 0:
                for i in range(self.popsize):
                    for j in range(self.dim):
                        # self.X_pop[i][j] = (self.upper_bound - self.lower_bound)*random.rand() + self.lower_bound   #两种初始化方法
                        self.X_pop[i][j] = random.rand()
                    self.best_mat[i][0] = self.object_function(self.X_pop[i,:])      # 最优值
                    self.best_mat[i][1: self.dim + 1] = self.X_pop[i,:]
            else:
                self.update_pop()
                for k in range(self.popsize):                 #更新bestmat
                    max_adapt = self.object_function(self.X_pop[k,:])
                    if max_adapt > self.best_mat[k][0]:
                        self.best_mat[k][0] = max_adapt
                        self.best_mat[k][1: self.dim + 1] = self.X_pop[k,:]

            for i in range(self.popsize):
                self.decpop[i] = self.object_function(self.X_pop[i,:])
            optimal = max(self.decpop)
            index = np.argwhere(self.decpop == optimal)
            self.best_y[0][generation] = optimal
            if generation == 0:
                self.variable[0][0] = optimal
                self.variable[0][1:self.dim+1] = self.X_pop[index[0][0],:]
            elif optimal > self.variable[0][0]:
                self.variable[0][0] = optimal
                self.variable[0][1:self.dim + 1] = self.X_pop[index[0][0], :]

        count = 0
        final_x = self.variable[0][1:self.dim + 1]
        for m in range(self.weights1.shape[0]):
            for n in range(self.weights1.shape[1]):
                self.weights1[m][n] = final_x[count]
                count += 1
        for o in range(self.bias1.shape[1]):
            self.bias1[0][o] = final_x[count]
            count += 1
        for p in range(self.weights2.shape[0]):
            for q in range(self.weights2.shape[1]):
                self.weights2[p][q] = final_x[count]
                count += 1
        for r in range(self.bias2.shape[1]):
            self.bias2[0][r] = final_x[count]
            count += 1

        # iter = linspace(0,self.T,num = self.T)[:,np.newaxis]
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.plot(iter, -self.best_y.T, 'b-')
        # plt.ion()  # 本次运行请注释，全局运行不要注释
        # plt.ioff()
        # plt.show()
        print(-self.best_y)



if __name__ == "__main__":
    # x = np.linspace(-1,1,num = 30)[:,np.newaxis]
    # y = np.fabs(np.sin(2 * np.pi * (x)))
    # noise = np.random.normal(0, 0.05, x.shape).astype(np.float32)
    # y = np.square(x)

    # x_test = np.linspace(-1,1,num = 100)[:,np.newaxis]
    # y_test = np.square(x_test)
    # y_test = np.fabs(np.sin(2 * np.pi * (x_test)))

    # x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    # y = np.array([[0],[1],[1],[0]])
    #
    # x_test = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,1,0],[1,1,0],[0,0,0],[1,0,0]])
    # y_test = np.array([[0],[1],[1],[0],[1],[0],[1],[0]])


    datafile = "C:/Users/MaxMai/Desktop/python/iris_test+train.xlsx"
    # datafile = "test.xlsx"
    x = excel_to_matrix(datafile, 0)
    y = excel_to_matrix(datafile, 3)

    x_test = excel_to_matrix(datafile, 1)
    y_test = excel_to_matrix(datafile, 2)

    NN = PSO_NeuralNetwork(x, y)
    NN.PSO_Optimizer()
    #
    iter = linspace(0, NN.T, num=NN.T)[:, np.newaxis]
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(iter, -NN.best_y.T, 'b-')
    plt.ion()  # 本次运行请注释，全局运行不要注释
    # plt.ioff()
    plt.xlabel("iteration")
    plt.ylabel("mean square error")
    NN.test(x_test, y_test)
    plt.show()

    iteration = 500
    loss = np.zeros([1,iteration])
    for j in range(iteration):
        NN.loss = 0
        NN.train()

        if j % 50 == 0:
            print(NN.loss)
        loss[0][j] = NN.loss


    iter = linspace(0,iteration,num = iteration)[:,np.newaxis]
    # fig = plt.figure()
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(iter,loss.T, 'b-')
    plt.ion()  # 本次运行请注释，全局运行不要注释
    plt.ioff()
    plt.xlabel("iteration")
    plt.ylabel("mean square error")
    NN.test(x_test, y_test)
    plt.show()

    # NN.test(x_test, y_test)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(x, y)
    # plt.ion()  # 本次运行请注释，全局运行不要注释
    # plt.show()

#     NN.test(array([[0,0,1]]),array([[0]]))
#     NN.test(array([[1, 0, 1]]), array([[1]]))
#     NN.test(array([[1, 1, 1]]), array([[0]]))
#     NN.test(array([[0, 0, 0]]), array([[0]]))
#     NN.test(array([[0, 1, 1]]), array([[1]]))   #predict
#     NN.test(array([[1, 1, 0]]), array([[0]]))   #predict
# NN.test(array([[0, 1]]), array([[0]]))
# NN.test(array([[1, 1]]), array([[1]]))
#     NN.test(array([[6,2.9, 4.5, 1.5]]), array([[0,1,0]]))


# print(loss)                             ##最好的误差

