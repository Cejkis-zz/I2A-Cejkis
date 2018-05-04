
import numpy as np

actions = ((-1, 0), (1, 0), (0, -1), (0, 1))

class Mapa:

    def __init__(self, mapa2, height, width):

        self.height = height
        self.width = width

        mapa = np.zeros((4, self.height, self.width))
        nrCilu = 0
        self.nrFinished = 0

        for i in range(len(mapa2)):
            for j in range(len(mapa2[0])):
                if mapa2[i][j] == "#":  # zed
                    mapa[3][i][j] = 1
                elif mapa2[i][j] == "$":  # kamen
                    mapa[1][i][j] = 1
                elif mapa2[i][j] == ".":  # cil
                    mapa[2][i][j] = 1
                    nrCilu += 1
                elif mapa2[i][j] == "@":  # hrac
                    mapa[0][i][j] = 1
                    hrac = [i, j]
                elif mapa2[i][j] == "+":  # hrac + cil
                    mapa[0][i][j] = 1
                    hrac = [i, j]
                    mapa[2][i][j] = 1
                    nrCilu += 1
                elif mapa2[i][j] == "*":  # kamen + cil
                    mapa[1][i][j] = 1
                    mapa[2][i][j] = 1
                    nrCilu += 1
                    self.nrFinished += 1

        self.map3D = mapa
        self.playerPos = hrac
        self.nrCilu = nrCilu

        # nahodne presun hrace
        self.map3D[0][self.playerPos[0]][self.playerPos[1]] = 0

        pos = self.playerPos
        for i in range(20):
            smer = actions[np.random.randint(0, 4)]
            newpos = [pos[0] + smer[0], pos[1] + smer[1]]
            if self.map3D[3][newpos[0]][newpos[1]] == 0 and self.map3D[1][newpos[0]][newpos[1]] == 0:
                pos = newpos
        self.playerPos = pos
        self.map3D[0][self.playerPos[0]][self.playerPos[1]] = 1


    def __getitem__(self, item):
        return self.map3D[item]

    def __setitem__(self, key, value):
        self.map3D[key] = value

    def printMap(self):
        for i in range(self.height):
            for j in range(self.width):
                if self.map3D[3][i][j] == 1: # zed
                    print('#',end='')
                elif self.map3D[1][i][j] == 1: # kameny
                    if self.map3D[2][i][j] == 1:
                        print('*', end='') # kamen + cil
                    else:
                        print('$',end='')
                elif self.map3D[2][i][j] == 1: # cile
                    if self.map3D[0][i][j] == 1: # cil+hrac
                        print('+',end='')
                    else:
                        print('.',end='')
                elif self.map3D[0][i][j] == 1: # hrac
                    print('@',end='')
                else:
                    print(' ', end='')
            print('\n', end = '')
