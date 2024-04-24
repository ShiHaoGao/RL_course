import numpy as np
import random
import os
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import keyMap as Map

class DQN():
    def __init__(self):
        self.n_input = Map.mapsize * Map.mapsize
        self.n_output = 1
        self.current_q_step = 0
        self.avg_loss = 0
        # placeholder是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。
        # 建立完session后，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        self.x = tf.placeholder("float", [None, Map.mapsize, Map.mapsize], name='x')
        self.y = tf.placeholder("float", [None, self.n_output], name='y')
        self.create_Q_network()
        self.create_training_method()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        # 它能让你在运行图的时候，插入一些计算图
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # tf.random_normal()函数用于从“服从指定正态分布的序列”中随机取出指定个数的值。  stddev: 正态分布的标准差
        wc1 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1), dtype=tf.float32, name='wc1')
        wc2 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.1), dtype=tf.float32, name='wc2')
        wc3 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.1), dtype=tf.float32, name='wc3')
        wd1 = tf.Variable(tf.random_normal([256, 128], stddev=0.1), dtype=tf.float32, name='wd1')
        wd2 = tf.Variable(tf.random_normal([128, self.n_output], stddev=0.1), dtype=tf.float32, name='wd2')
        # tf.Variable 得到的是张量，而张量并不是具体的值，而是计算过程
        bc1 = tf.Variable(tf.random_normal([64], stddev=0.1), dtype=tf.float32, name='bc1')
        bc2 = tf.Variable(tf.random_normal([128], stddev=0.1), dtype=tf.float32, name='bc2')
        bc3 = tf.Variable(tf.random_normal([256], stddev=0.1), dtype=tf.float32, name='bc3')
        bd1 = tf.Variable(tf.random_normal([128], stddev=0.1), dtype=tf.float32, name='bd1')
        bd2 = tf.Variable(tf.random_normal([self.n_output], stddev=0.1), dtype=tf.float32, name='bd2')

        weights = {
            'wc1': wc1,
            'wc2': wc2,
            'wc3': wc3,
            'wd1': wd1,
            'wd2': wd2
        }

        biases = {
            'bc1': bc1,
            'bc2': bc2,
            'bc3': bc3,
            'bd1': bd1,
            'bd2': bd2
        }

        self.Q_value = self.conv_basic(self.x, weights, biases)
        self.Q_Weihgts = [weights, biases]

    def conv_basic(self, _input, _w, _b):
        # input
        _out = tf.reshape(_input, shape=[-1, Map.mapsize, Map.mapsize, 1])
        # conv layer 1  conv2d 用于做二维卷积  strides, # 步长参数  padding, # 卷积方式
        _out = tf.nn.conv2d(_out, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
        # bias_add 一个叫bias的向量加到一个叫value的矩阵上，是向量与矩阵的每一行进行相加
        _out = tf.nn.relu(tf.nn.bias_add(_out, _b['bc1']))
        # ksize 池化窗口的大小，取一个四维向量  padding： 填充的方法，SAME或VALID，SAME表示添加全0填充，VALID表示不添加
        _out = tf.nn.max_pool(_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # conv layer2
        _out = tf.nn.conv2d(_out, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
        _out = tf.nn.relu(tf.nn.bias_add(_out, _b['bc2']))
        _out = tf.nn.max_pool(_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # conv layer3
        _out = tf.nn.conv2d(_out, _w['wc3'], strides=[1, 1, 1, 1], padding='SAME')
        _out = tf.nn.relu(tf.nn.bias_add(_out, _b['bc3']))
        # 计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
        _out = tf.reduce_mean(_out, [1, 2])
        # fully connected layer1 matmul 两个矩阵中对应元素各自相乘
        _out = tf.nn.relu(tf.add(tf.matmul(_out, _w['wd1']), _b['bd1']))
        # fully connected layer2
        _out = tf.add(tf.matmul(_out, _w['wd2']), _b['bd2'])
        return _out

    def create_training_method(self):
        # squared_difference 计算张量 x、y 对应元素差平方
        self.cost = tf.reduce_mean(tf.squared_difference(self.Q_value, self.y))
        self.optm = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam').minimize(self.cost)

    def restore(self):
        if os.path.exists('Saver/cnnsaver.ckpt-0.index'):
            self.saver.restore(self.sess, os.path.abspath('Saver/cnnsaver.ckpt-0'))
            print("加载训练权重***********")

    def computerPlay(self, IsTurnWhite):
        if IsTurnWhite:
            print('白旗走')
            # 如果该白旗走的话 用黑的棋盘，1代表黑，-1代表白
            board = np.array(Map.blackBoard)
        else:
            print('黑旗走')
            # 如果该黑旗走的话 用白的棋盘 1代表白，-1代表黑
            board = np.array(Map.whiteBoard)
        # 建立所有可下位置的数组，每下一个位置一个数组
        boards = []
        # 当前棋谱中空白的地方
        positions = []
        for i in range(Map.mapsize):
            for j in range(Map.mapsize):
                # 如果这个当前棋谱这个位置是空白的
                if board[j][i] == Map.backcode:
                    predx = np.copy(board)
                    # -1代表自己，更方便计算
                    predx[j][i] = -1
                    boards.append(predx)
                    positions.append([i, j])
        if len(positions) == 0:
            return 0, 0, 0
        # 计算所有可下的位置的价值
        nextStep = self.sess.run(self.Q_value, feed_dict={self.x: boards})
        maxx = 0
        maxy = 0
        maxValue = -1000  # 实际最大价值  用于后续学习
        # 从所有可下的地方找一个价值最大的位置下棋
        for i in range(len(positions)):
            value = nextStep[i] + random.randint(0, 10) / 1000  # 如果没有最优步子 则随机选择一步
            if value > maxValue:
                maxValue = value
                maxx = positions[i][0]
                maxy = positions[i][1]
        print(str(maxx) + ',' + str(maxy))
        print('此位置的价值为：' + str(maxValue[0]))
        return maxx, maxy, maxValue

    # 下完了一局就更新一下AI模型
    def TrainOnce(self, winner):
        # 记录棋图
        print("下完了一局就更新一下AI模型")
        # board1 白棋 board2 黑棋
        board1 = np.array(Map.mapRecords1)
        board2 = np.array(Map.mapRecords2)
        # 记录棋步
        step1 = np.array(Map.stepRecords1)
        step2 = np.array(Map.stepRecords2)
        # 记录得分
        scoreR1 = np.array(Map.scoreRecords1)
        scoreR2 = np.array(Map.scoreRecords2)
        board1 = np.reshape(board1, [-1, Map.mapsize, Map.mapsize])
        board2 = np.reshape(board2, [-1, Map.mapsize, Map.mapsize])
        step1 = np.reshape(step1, [-1, Map.mapsize, Map.mapsize])
        step2 = np.reshape(step2, [-1, Map.mapsize, Map.mapsize])

        score1 = []
        score2 = []

        board1 = (board1 * (1 - step1)) + step1 * Map.blackcode
        board2 = (board2 * (1 - step2)) + step2 * Map.blackcode
        # 每步的价值 = 奖励（胜1 负-0.9） + 对方棋盘能达到的最大价值（max taget Q） * （-0.9）
        for i in range(len(board1)):
            if i == len(scoreR2):  # 白方已经五连  白方赢
                print('白方已经五连，白方赢')
                score1.append([1.0])  # 白方的最后一步获得1分奖励
            else:
                # 白方的价值为：黑方棋盘能达到的最大价值（max taget Q） * （-0.9）
                score1.append([scoreR2[i][0] * -0.9])
        if winner == 2:
            print('惩罚白方的最后一步，将其价值设为 -0.9')
            score1[len(score1) - 1][0] = -0.9

        # 1 白棋 2 黑棋
        for i in range(len(board2)):
            if i == len(scoreR1) - 1:  # 黑方赢
                print('黑方已经五连，黑方赢')
                score2.append([1.0])
            else:
                # 黑棋的得分为：白方棋盘能达到的最大价值（max taget Q） * （-0.9）
                score2.append([scoreR1[i + 1][0] * -0.9])
        if winner == 1:
            print('惩罚黑方的最后一步，将其价值设为 -0.9')
            # 惩罚黑方的最后一步
            score2[len(score2) - 1][0] = -0.9

        # 一次完成多个数组的拼接
        borders = np.concatenate([board1, board2], axis=0)
        scores = np.concatenate([score1, score2], axis=0)
        _, totalLoss = self.sess.run([self.optm, self.cost], feed_dict={self.x: borders,
                                                                        self.y: scores})
        self.avg_loss += totalLoss
        print('train avg loss ' + str(self.avg_loss))
        self.avg_loss = 0
        # os.path.abspath取决于os.getcwd,如果是一个绝对路径，就返回，
        # 如果不是绝对路径，根据编码执行getcwd/getcwdu.然后把path和当前工作路径连接起来
        self.saver.save(self.sess, os.path.abspath('test-Saver/cnnsaver.ckpt'), global_step=1)

    def PlayWidthHuman(self):
        # 读取历史存储的模型
        self.restore()
        Map.PlayWithComputer = self.computerPlay
        # Map.TrainNet = self.TrainOnce
        # Map.ShowWind()


if __name__ == '__main__':
    dqn = DQN()
    dqn.PlayWidthHuman()
