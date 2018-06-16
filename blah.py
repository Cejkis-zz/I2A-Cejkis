

for i in range(30000):

    f = open("/home/cejkis/SokoGen/sokohard/levels111/output" + str(i) + ".sok")

    a = f.readlines()
    for j in a:
        print(j[:-1])

    f.close()



