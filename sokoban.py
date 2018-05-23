import copy

from numpy import zeros
import numpy as np
import random
from map import Mapa

sizes = [111,121,122,1229,1239, 222]
STATE_SIZE = (4, 8, 5)
MAPSIZE = [363]

def decideParameters(MAPSIZE = 1239):
    STATE_SIZE = (4, 8, 5)
    MAX_MOVES = 120

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

    if MAPSIZE == 222:
        STATE_SIZE = (4, 8, 8)
        MAX_MOVES = 120
        MAPS = 640000

    if MAPSIZE == 363:
        STATE_SIZE = (4, 8, 5)
        MAX_MOVES = 120
        MAPS = 640000

    return STATE_SIZE,MAX_MOVES,MAPS

actions = ((-1,0),(1,0),(0,-1),(0,1))

class Sokoban:

    def __init__(self):
        global MAPSIZE
        self.it = 0
        self.STATE_SIZE, self.MAX_MOVES, self.MAPS = decideParameters(MAPSIZE[self.it])

        f = open("./levels/" + str(MAPSIZE[self.it]))
        self.allmaps = f.readlines()

        f.close()

        self.nrofmaps = len(self.allmaps) # todo check if module 8 == 0

        self.reset()

    def reset(self):
        global MAPSIZE
        self.moves = 0

        if len(MAPSIZE) > 1:
            self.it = (self.it + 1) % len(MAPSIZE)
            self.STATE_SIZE, self.MAX_MOVES, self.MAPS = decideParameters(MAPSIZE[self.it])

        mapNr = random.randint(0, self.nrofmaps / 8 - 1)

        map2D = self.allmaps[mapNr * 8:(mapNr + 1) * 8]

        self.mapaObjekt = Mapa(map2D, STATE_SIZE[1], STATE_SIZE[2])

        return self.mapaObjekt.map3D

    def newImagination(self):
        self.imagMap = copy.deepcopy(self.mapaObjekt)

    def doImaginaryAction(self, action):
        return self.doAction(self.imagMap, action)

    def step(self, action):
        self.moves += 1
        map, reward, done = self.doAction(self.mapaObjekt, action)
        return map,reward, self.moves >= self.MAX_MOVES or done

    def doAction(self, map, action03): # returns new map and a reward

        action = actions[action03] # maps action from 0-3 to (+-1,+-1)

        newPos = map.playerPos[0] + action[0], map.playerPos[1] + action[1]
        movedbrick = False
        reward = -0.01  # negativni odmena za krok

        if map[1][newPos[0]][newPos[1]]: # pokud tam je kostka
            newBrickPos = newPos[0] + action[0], newPos[1] + action[1] # nova pozice kostky
            if map[1][newBrickPos[0]][newBrickPos[1]] == 0 and map[3][newBrickPos[0]][newBrickPos[1]] == 0 : # a za ni neni zed ani dalsi kostka,

                # posunu kostku
                map[1][newBrickPos[0]][newBrickPos[1]] = 1
                map[1][newPos[0]][newPos[1]] = 0

                # a udelam krok
                map[0][map.playerPos[0]][map.playerPos[1]] = 0
                map[0][newPos[0]][newPos[1]] = 1
                movedbrick = True

                map.playerPos = newPos

        elif map[3][newPos[0]][newPos[1]]==0: # pokud tam neni zed, udelam krok
            map[0][map.playerPos[0]][map.playerPos[1]] = 0
            map[0][newPos[0]][newPos[1]] = 1

            map.playerPos = newPos

        done = False

        if movedbrick:
            if map[2][newPos[0]][newPos[1]]: # posunul jsem z cile
                reward += -0.1
                map.nrFinished -= 1
            if map[2][newBrickPos[0]][newBrickPos[1]]: # posunul jsem na cil
                reward += 0.1
                map.nrFinished += 1
                if map.nrCilu == map.nrFinished:
                    reward += 1
                    done = True

        return map.map3D, reward, done
