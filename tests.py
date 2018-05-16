
import random
import sokoban

env = sokoban.Sokoban()

score = 0
step = 0

while True:

    #env.mapaObjekt.printMap()
    a = random.randint(0,3)
    #a = int(input("next step"))
    s,r,d = env.step(a)

    if d:
        step += 1
        env.reset()

        if r > 0.5:
            score += 1

        print(score / step)
