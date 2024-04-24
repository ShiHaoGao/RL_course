import copy

import numpy as np


class QuantumBoardEnv:
    def __init__(self):
        # 初始化棋盘状态，0代表没有子，1是黑子，-1是白子
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 黑子先下

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 黑子先下
        return self.state_to_int(self.board)  # 把3*3棋盘返回一个数字

    def step(self, action):
        x, y = divmod(action, 3)  # 因为action是Q值表里面的所有的动作的一维序号，所以除以三的商和余数分别代表了行数和列数
        if self.board[x, y] != 0:
            print("无效步，下一个epoch")
            return self.board.flatten(), 0, True, {}  # 如果下的地方不为0，即有子了，所以这是一个无效步
        self.board[x, y] = self.current_player  # 下的地方置为黑子
        reward, done = self.check_game_end(x, y)
        self.current_player = 0 - self.current_player  # 交换黑白子，黑子下完白子下
        return self.state_to_int(self.board), reward, done, {}

    def check_game_end(self, x, y):
        # 判断棋局结束: 行、列、对角线连成三个子
        line_sum = 3 * self.current_player
        if (np.sum(self.board[x, :]) == line_sum or  # 行
                np.sum(self.board[:, y]) == line_sum or  # 列
                np.sum(np.diag(self.board)) == line_sum):  # 对角线
            return 1 if self.current_player == 1 else -1, True  # 如果黑子赢了，返回奖励1；如果白子赢了，返回奖励-1
        if not np.any(self.board == 0):
            return 0, True  # 中间不设奖励
        return 0, False

    def state_to_int(self, state):
        # 把这个棋盘变成一个数字；把棋盘展开看作是3进制的数转换成10进制数
        self.s = copy.deepcopy(state)
        self.s[self.s == -1] = 2
        flat_state = int(''.join(str(x) for row in self.s for x in row), 3)
        return flat_state

    def render(self):
        # 棋盘展示
        symbols = {0: ".", 1: "B", -1: "W"}
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print()

    @property
    def action_space(self):
        return np.flatnonzero(self.board == 0)  # 返回棋盘上值为0的索引序列

    @property
    def observation_space(self):
        return {'shape': (1,), 'type': 'int', 'high': 3 ** 9 - 1, 'low': 0}


# Example of usage
if __name__ == "__main__":
    env = QuantumBoardEnv()
    env.reset()
    env.render()
    state, reward, done, info = env.step(4)
    env.render()
    state, reward, done, info = env.step(0)
    env.render()
