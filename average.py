


f = open('workfile', 'r')

i = 0
sum = 0

for line in f:

        line = line.replace(',','.')

        if i%5 ==0:
            print(("%.3f" % (sum/5)).replace('.',','))
            i=0
            sum = 0

        i=i+1
        sum += float(line)
