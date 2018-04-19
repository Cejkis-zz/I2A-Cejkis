
from numpy import zeros
import numpy as np
import random
from Mapa import Mapa

sizes = [111,121,122,1229,1239]
STATE_SIZE = (4, 8, 5)

def decideParameters(MAPSIZE = 1239):
    STATE_SIZE = (4, 8, 5)
    MAX_MOVES = 80

    if MAPSIZE == 121:
        MAPS = 160000

    if MAPSIZE == 122:
        MAPS = 75000

    if MAPSIZE == 1229: # bez --changes
        MAPS = 50000

    if MAPSIZE == 1239: # bez --changes
        MAPS = 64000

    if MAPSIZE == 111:
        STATE_SIZE = (4,5,5)
        MAX_MOVES = 50
        MAPS = 100000

    return STATE_SIZE,MAX_MOVES,MAPS

actions = ((-1,0),(1,0),(0,-1),(0,1))

class Sokoban:

    def __init__(self):
        self.small = True
        self.map3D = self.reset()

    def reset(self):
        self.moves = 0
        
        if self.small:
            self.small = False
            MAPSIZE = 111
        else:
            self.small = True
            MAPSIZE = 1229
            
        self.STATE_SIZE, self.MAX_MOVES, self.MAPS = decideParameters(MAPSIZE)
        
        mapNr = random.randint(0, self.MAPS-1)
        #print(mapNr)

        with open("/home/cejkis/SokoGen/sokohard/levels" + str(MAPSIZE) + "/output" + str(mapNr) + ".sok") as f:
            mapa = f.readlines()
            map2D = []
            for i in mapa:
                map2D.append(i[:len(i) - 1])

        self.map3D = Mapa(map2D, 5, 8)

        # nahodne presun hrace
        self.map3D[0][self.playerPos[0]][self.playerPos[1]] = 0

        pos = self.playerPos
        for i in range(20):
            smer = actions[random.randint(0,3)]
            newpos = [pos[0] + smer[0], pos[1] + smer[1]]
            if self.map3D[0][newpos[0]][newpos[1]] == 0 and self.map3D[3][newpos[0]][newpos[1]] == 0:
                pos = newpos
        self.playerPos = pos
        self.map3D[0][self.playerPos[0]][self.playerPos[1]] = 1

        return self.map3D


    def newImagination(self):
        self.imagMap = np.copy(map)    # todo kopiruje tohle? a zkopiruj i pocet splnenych cihel

    def doImaginaryAction(self, action):
        return self.doAction(self.imagMap, action)

    def step(self, action):
        self.moves += 1
        map, reward, done = self.doAction(self.map3D, action)
        return map,reward, self.moves >= self.MAX_MOVES or done

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

        done = False

        if movedbrick:
            if map[2][newPos[0]][newPos[1]]: # posunul jsem z cile
                reward += -0.1
                self.nrFinished -= 1
            if map[2][newBrickPos[0]][newBrickPos[1]]: # posunul jsem na cil
                reward += 0.1
                self.nrFinished += 1
                if self.nrCilu == self.nrFinished:
                    reward += 1
                    done = True

        return map, reward, done
