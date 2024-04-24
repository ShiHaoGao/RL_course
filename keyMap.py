import tkinter as tk
import os
import time
import copy

# 定义窗口
# top = tk.Tk()
# top.title("AI自动玩五子棋")
# top.geometry('400x300')

# 定义地图尺寸
mapsize = 4

# 元素尺寸
pixsize = 20

# 连子个数
winSet = 3

# 空白编号
backcode = 0
# 白棋
whitecode = 1
# 黑棋
blackcode = -1

# 定义画布
# canvas = tk.Canvas(top, height=mapsize * pixsize, width=mapsize * pixsize,
#                    bg="gray")
# canvas.pack(pady=25)

# for i in range(mapsize):
#     canvas.create_line(i * pixsize, 0,
#                        i * pixsize, mapsize * pixsize,
#                        fill='black')
#     canvas.create_line(0, i * pixsize,
#                        mapsize * pixsize, i * pixsize,
#                        fill='black')

# 初始棋盘
whiteBoard = []
stepBoard = []
for i in range(mapsize):
    row = []
    rowBak = []
    for j in range(mapsize):
        row.append(0)
        rowBak.append(backcode)
    whiteBoard.append(rowBak)
    stepBoard.append(row)
blackBoard = copy.deepcopy(whiteBoard)

# 棋子列表
childMap = []

# 记录棋图
mapRecords1 = []
mapRecords2 = []

# 记录棋步
stepRecords1 = []
stepRecords2 = []
# 记录得分
scoreRecords1 = []
scoreRecords2 = []

isGameOver = False

IsTurnWhite = True


def Restart():
    global isGameOver
    global IsTurnWhite
    # for child in childMap:
    #     canvas.delete(child)
    childMap.clear()
    isGameOver = False
    IsTurnWhite = True
    mapRecords1.clear()
    mapRecords2.clear()
    stepRecords1.clear()
    stepRecords2.clear()
    scoreRecords1.clear()
    scoreRecords2.clear()
    for i in range(mapsize):
        for j in range(mapsize):
            whiteBoard[j][i] = backcode
            blackBoard[j][i] = backcode


WinDataSetPath = 'DataSets\\win'
LosDataSetPath = 'DataSets\\los'

TrainNet = None


def SaveDataSet(tag):
    print("savedataset")
    if TrainNet != None:
        print("if TrainNet != None:")
        TrainNet(tag)
    else:
        winfilename = WinDataSetPath + '\\' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + '.txt'
        losfilename = LosDataSetPath + '\\' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + '.txt'
        if not os.path.exists('DataSets'):
            os.mkdir('DataSets')
        if not os.path.exists(WinDataSetPath):
            os.mkdir(WinDataSetPath)
        if not os.path.exists(LosDataSetPath):
            os.mkdir(LosDataSetPath)
        strInfo1 = ''
        for i in range(len(mapRecords1)):
            for j in range(mapsize):
                for k in range(mapsize):
                    strInfo1 += str(mapRecords1[i][j][k]) + ','
            strInfo1 += '\n'
            for j in range(mapsize):
                for k in range(mapsize):
                    strInfo1 += str(stepRecords1[i][j][k]) + ','
            strInfo1 += '\n'
        strInfo2 = ''
        for i in range(len(mapRecords2)):
            for j in range(mapsize):
                for k in range(mapsize):
                    strInfo2 += str(mapRecords2[i][j][k]) + ','
            strInfo2 += '\n'
            for j in range(mapsize):
                for k in range(mapsize):
                    strInfo2 += str(stepRecords2[i][j][k]) + ','
            strInfo2 += '\n'
        if tag == 1:
            with open(winfilename, "w") as f:
                f.write(strInfo1)
            with open(losfilename, "w") as f:
                f.write(strInfo2)
        else:
            with open(winfilename, "w") as f:
                f.write(strInfo2)
            with open(losfilename, "w") as f:
                f.write(strInfo1)


# 判断是否练成了五条线
def JudgementResult():
    global isGameOver
    judgemap = whiteBoard
    for i in range(mapsize):
        for j in range(mapsize):
            if judgemap[j][i] != backcode:
                tag = judgemap[j][i]
                checkrow = True
                checkCol = True
                checkLine = True
                checkLine2 = True
                for k in range(winSet - 1):
                    if i + k + 1 < mapsize:  # 行
                        if (judgemap[j][i + k + 1] != tag) and checkrow:
                            checkrow = False
                        if j + k + 1 < mapsize:  # 斜线
                            if (judgemap[j + k + 1][i + k + 1] != tag) and checkLine:
                                checkLine = False
                        else:
                            checkLine = False
                    else:
                        checkrow = False
                        checkLine = False
                    if j + k + 1 < mapsize:  # 列
                        if (judgemap[j + k + 1][i] != tag) and checkCol:
                            checkCol = False
                        if i - k - 1 >= 0:  # 斜线
                            if (judgemap[j + k + 1][i - k - 1] != tag) and checkLine2:
                                checkLine2 = False
                        else:
                            checkLine2 = False
                    else:
                        checkCol = False
                        checkLine2 = False
                    if not checkrow and not checkCol and not checkLine and not checkLine2:
                        break
                if checkrow or checkCol or checkLine or checkLine2:
                    print("结束了!!!!!!!!")
                    isGameOver = True
                    # SaveDataSet(tag)
                    return tag
    return 0


PlayWithComputer = None

GetMaxScore = None


def playChess(event):
    if isGameOver:
        print('game is over, restart!')
        Restart()
        return
    x = event.x // pixsize
    y = event.y // pixsize
    if x >= mapsize or y >= mapsize:
        return
    if whiteBoard[y][x] != backcode:
        print("此处已经有棋子了！！！")
        return
    score = 0
    if PlayWithComputer != None:
        _x, _y, score = PlayWithComputer(IsTurnWhite)
    
    res = chess(x, y, score)
    if res == 0:
        if PlayWithComputer != None:
            x, y, score = PlayWithComputer(IsTurnWhite)
            res = chess(x, y, score)


def KeyChess(x,y,score):
    if x >= mapsize or y >= mapsize:
        print("超过接线")
        return 
    if whiteBoard[y][x] != backcode:
        print("此处已经有棋子了！！！")
        return 
    score = 0
    if PlayWithComputer != None:
        # print("current status",IsTurnWhite )
        _x, _y, score = PlayWithComputer(IsTurnWhite)
        # print("对面现在哪儿了：",_x,_y)
    res = chess(x, y, score)
    if res == 0:
        if PlayWithComputer != None:
            # print("current status",IsTurnWhite )
            x_new, y_new, score = PlayWithComputer(IsTurnWhite)
            res = chess(x_new, y_new, score)
            # print("对面现在哪儿了：",x,y)
            return x_new,y_new
    return -1,-1
     




def chess(x, y, score):
    global IsTurnWhite
    if isGameOver:
        print('game is over, restart!')
        Restart()
        return -1
    if whiteBoard[y][x] != backcode:
        print("在chess中此处已经有棋子了!!!!!!!!")
        Restart()
        return -1
    step = copy.deepcopy(stepBoard)
    step[y][x] = 1
    if IsTurnWhite:  # 白棋是人工走的 如果过用来当训练集 用反转棋盘
        mapRecords1.append(copy.deepcopy(blackBoard))
        stepRecords1.append(step)
        scoreRecords1.append(score)
        whiteBoard[y][x] = whitecode  # 1白 -1黑
        blackBoard[y][x] = blackcode
        # child = canvas.create_oval(x * pixsize,
        #                            y * pixsize,
        #                            x * pixsize + pixsize,
        #                            y * pixsize + pixsize, fill='white')
    else:
        mapRecords2.append(copy.deepcopy(whiteBoard))
        stepRecords2.append(step)
        scoreRecords2.append(score)
        whiteBoard[y][x] = blackcode  # 1白 -1黑
        blackBoard[y][x] = whitecode
        # child = canvas.create_oval(x * pixsize,
        #                            y * pixsize,
        #                            x * pixsize + pixsize,
        #                            y * pixsize + pixsize, fill='black')
    IsTurnWhite = not IsTurnWhite
    # childMap.append(child)
    return JudgementResult()



# 按钮的点击事件
def AutoPlayOnce():
    if PlayWithComputer != None:
        x, y, score = PlayWithComputer(IsTurnWhite)
        chess(x, y, score)


# btnAuto = tk.Button(top, text="重新开始或者自动走1次", command=AutoPlayOnce)
# btnAuto.pack()
# 画布与鼠标左键进行绑定
# canvas.bind("<B1-Motion>", playChess)
# canvas.bind("<Button-1>", playChess)


# 按钮的点击事件
def AutoPlayOne():
    global isGameOver
    if PlayWithComputer != None:
        for i in range(222):
            if isGameOver:
                break
            x, y, score = PlayWithComputer(IsTurnWhite)
            chess(x, y, score)

# btnAuto = tk.Button(top, text="自动玩一局", command=AutoPlayOne)
# btnAuto.pack()
# canvas.bind("<Button-2>", playChess)



# 显示游戏窗口
def ShowWind():
    play_game()
    # top.mainloop()


def play_game():
    global isGameOver, IsTurnWhite
    while not isGameOver:
        print_board(whiteBoard)  # 显示当前棋盘的状态
        try:
            # 这里的x,y是玩家输入的坐标
            x, y = map(int, input(f"轮到{'白棋' if IsTurnWhite else '黑棋'}，请输入您的落子坐标 (格式：x y): ").split())


        except ValueError:
            print("无效的输入，请输入两个数字")
            continue

        if not (0 <= x < mapsize and 0 <= y < mapsize and whiteBoard[y][x] == backcode):
            print("非法落子，请重新输入")
            continue

        # 这里的x_new和y_new是对面下的位置
        x_new , y_new = KeyChess(x, y, 0)


        print("对面走哪儿了:::::::",x_new , y_new)

def dqn_input( x, y ):
    global isGameOver

    if not (0 <= x < mapsize and 0 <= y < mapsize and whiteBoard[y][x] == backcode):
        print("非法落子，请重新输入")

        # 这里的x_new和y_new是对面下的位置

    x_new , y_new = KeyChess(x, y, 0)
    is_dqn_done = isGameOver
    return x_new , y_new , is_dqn_done


def print_board(board):
    for row in board:
        print(' '.join(['.' if x == backcode else 'W' if x == whitecode else 'B' for x in row]))
