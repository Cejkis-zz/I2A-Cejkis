import copy

from numpy import zeros
import numpy as np
import random
from map import Mapa

#sizes = [111,121,122,1229,1239, 222]
STATE_SIZE = (4, 8, 5)
MAPSIZE = [363]

def decideParameters(MAPSIZE = 1239):
    STATE_SIZE = (4, 8, 5)
    MAX_MOVES = 120

    if MAPSIZE == 363:
        STATE_SIZE = (4, 8, 5)
        MAX_MOVES = 120

    if MAPSIZE == 331:
        STATE_SIZE = (4, 5, 5)
        MAX_MOVES = 70

    return STATE_SIZE,MAX_MOVES

actions = ((-1,0),(1,0),(0,-1),(0,1))

class Sokoban:

    def __init__(self):
        global MAPSIZE
        self.it = 0
        self.STATE_SIZE, self.MAX_MOVES = decideParameters(MAPSIZE[self.it])

        f = open("./levels/" + str(MAPSIZE[self.it]))
        self.allmaps = f.readlines()

        f.close()

        self.nrofmaps = len(self.allmaps) # todo check if  %8 == 0

        self.reset()

    def reset(self):
        global MAPSIZE
        self.moves = 0

        if len(MAPSIZE) > 1:
            self.it = (self.it + 1) % len(MAPSIZE)
            self.STATE_SIZE, self.MAX_MOVES, self.MAPS = decideParameters(MAPSIZE[self.it])

        height = 8
        mapNr = random.randint(0, self.nrofmaps / height - 1) #
        #print(mapNr)
        map2D = self.allmaps[mapNr * height:(mapNr + 1) * height]

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
