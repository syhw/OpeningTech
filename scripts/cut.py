
import sys, random

random.seed()
lPvP = open("lPvP.txt", 'w')
tPvP = open("tPvP.txt", 'w')
for line in sys.stdin:
    if (random.random() < 0.1):
        tPvP.write(line)
    else:
        lPvP.write(line)

lPvP.close()
tPvP.close()

