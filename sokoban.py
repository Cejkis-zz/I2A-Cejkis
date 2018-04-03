
from numpy import zeros
import random


MAX_MOVES = 50
MAPS = 160000

actions = ((-1,0),(1,0),(0,-1),(0,1))

class Sokoban:

    class dummy:
        def __init__(self):
            self.n = 4

    def __init__(self):
        self.map3D = self.reset()

    def reset(self):
        self.moves = 0

        mapNr = random.randint(0, MAPS-1)
        #print(mapNr)

        with open("/home/cejkis/PycharmProjects/I2A-Cejkis/levels1/levels121/output" + str(mapNr) + ".sok") as f:
            mapa = f.readlines()
            map2D = []
            for i in mapa:
                map2D.append(i[:len(i) - 1])

        self.width = len(map2D[0])
        self.height = len(map2D)

        #for i in map2D:
        #    print(i)
        self.processMap(map2D)
        #self.printMap()

        return self.map3D

    def printMap(self):
        for i in range(self.height):
            for j in range(self.width):
                if self.map3D[3][i][j] == 1: # zed
                    print('#',end='')
                elif self.map3D[1][i][j] == 1: # kameny
                    print('$',end='')
                elif self.map3D[2][i][j] == 1: # cile
                    if self.map3D[0][i][j] == 1:
                        print('+',end='')
                    else:
                        print('.',end='')
                elif self.map3D[0][i][j] == 1: # hrac
                    print('@',end='')
                else:
                    print(' ', end='')
            print('\n', end = '')

    def processMap(self, mapa2):  # 0: hrac, 1:kameny, 2:cile 3:zed
        mapa = zeros((4, self.height, self.width))
        nrCilu = 0
        self.nrFinished = 0

        for i in range(self.height):
            for j in range(self.width):
                if mapa2[i][j] == "#": # zed
                    mapa[3][i][j] = 1
                elif mapa2[i][j] == "$": # kameny
                    mapa[1][i][j] = 1
                elif mapa2[i][j] == ".": # cile
                    mapa[2][i][j] = 1
                    nrCilu +=1
                elif mapa2[i][j] == "@": # hrac
                    mapa[0][i][j] = 1
                    hrac = [i, j]
                elif mapa2[i][j] == "+": # hrac + cil
                    mapa[0][i][j] = 1
                    hrac = [i, j]
                    mapa[2][i][j] = 1
                    nrCilu += 1
        self.map3D = mapa
        self.playerPos = hrac
        self.nrCilu = nrCilu

    def newImagination(self):
        self.imagMap = map    # todo kopiruje tohle? a zkopiruj i pocet splnenych cihel

    def doImaginaryAction(self, action):
        self.doAction(self.imagMap, action)

    def step(self, action):
        self.moves += 1
        map, reward, done = self.doAction(self.map3D, action)
        return map,reward, self.moves >= MAX_MOVES or done

    def doAction(self, map, action03): # returns new map and a reward

        action = actions[action03] # maps action from 0-3 to (+-1,+-1)

        newPos = self.playerPos[0] + action[0], self.playerPos[1] + action[1]
        movedbrick = False
        reward = -0.01  # negativni odmena za krok

        if map[1][newPos[0]][newPos[1]]: # pokud tam je kostka
            newBrickPos = newPos[0] + action[0], newPos[1] + action[1] # nova pozice kostky
            if map[1][newBrickPos[0]][newBrickPos[1]] == 0 and map[3][newBrickPos[0]][newBrickPos[1]] == 0 : # a za ni neni zed ani dalsi kostka,

                # posunu kostku
                map[1][newBrickPos[0]][newBrickPos[1]] = 1
                map[1][newPos[0]][newPos[1]] = 0

                # a udelam krok
                map[0][self.playerPos[0]][self.playerPos[1]] = 0
                map[0][newPos[0]][newPos[1]] = 1
                movedbrick = True

                self.playerPos[0] += action[0]
                self.playerPos[1] += action[1]

        elif map[3][newPos[0]][newPos[1]]==0: # pokud tam neni zed, udelam krok
            map[0][self.playerPos[0]][self.playerPos[1]] = 0
            map[0][newPos[0]][newPos[1]] = 1

            self.playerPos[0] += action[0]
            self.playerPos[1] += action[1]

        else:
            reward = -0.01  # negativni odmena za krok do zdi

        done = False

        if movedbrick:
            if map[2][newPos[0]][newPos[1]]: # posunul jsem z cile
                reward += -0.1
                self.nrFinished -= 1
            if map[2][newBrickPos[0]][newBrickPos[1]]: # posunul jsem na cil
                reward += 0.
                self.nrFinished += 1
                if self.nrCilu == self.nrFinished:
                    reward += 1
                    done = True

        return map, reward, done
