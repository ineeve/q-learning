import numpy as np
import matplotlib.pyplot as plt
import random
SIZE = 4
rewards = np.zeros((SIZE,SIZE))
rewards[3][3] = 1
rewards[1][1] = -1
rewards[2][2] = -1

plt.imshow(rewards, cmap='binary')
plt.show()
actions = ['u','r','d','l']

def selectBestAction(actions):
    best = (actions[0],0)
    for index,a in enumerate(actions):
        if (a[1] > best[1]):
            best = (a,index)
    return best

def selectAction(actions, epsilon):
    if (random.random() > epsilon):
        return selectBestAction(actions)
    else:
        rd = random.randint(0,len(actions)-1)
        return (actions[rd], rd)

def printMatrix(m, size):
    for i in range(size):
        for j in range(size):
            print('[', end='')
            for a in m[i][j]:
                print(a[0] + "=" + "%.2f" % a[1], end=';')
            print(']', end='')
        print('')
    plt.imshow(rewards, cmap='binary')
    plt.show()

def qLearn():
    alpha = 0.9
    epsilon = 0.1
    discount = 0.5
    q = [
        [[['r',0],['d',0]], [['r',0],['d',0],['l',0]], [['r',0],['d',0],['l',0]], [['d',0],['l',0]]],
        [[['u',0],['r',0],['d',0]], [['u',0],['r',0],['d',0],['l',0]], [['u',0],['r',0],['d',0],['l',0]], [['u',0],['d',0],['l',0]]],
        [[['u',0],['r',0],['d',0]], [['u',0],['r',0],['d',0],['l',0]], [['u',0],['r',0],['d',0],['l',0]], [['u',0],['d',0],['l',0]]],
        [[['u',0],['r',0]], [['u',0],['r',0],['l',0]], [['u',0],['r',0],['l',0]], [['u',0],['l',0]]]
        ]
    for episode in range(0,100):
        l=0
        c=0
        while(l != 3 or c != 3):
            actions = q[l][c]
            actionTuple = selectAction(actions, epsilon)
            direction = actionTuple[0][0]
            # take action
            nextC = c
            nextL = l
            if (direction == 'r'): nextC = c+1
            elif (direction == 'd'): nextL = l+1
            elif (direction == 'l'): nextC = c-1
            elif (direction == 'u'): nextL = l-1
            reward = rewards[nextL][nextC]
            nextActions = q[nextL][nextC]
            nextBestActionValue = selectBestAction(nextActions)[0][1]
            prevStateValue = q[l][c][actionTuple[1]][1]
            q[l][c][actionTuple[1]][1] += alpha * (reward + discount* nextBestActionValue - prevStateValue)
            l = nextL
            c = nextC
    printMatrix(q, SIZE)
qLearn()

