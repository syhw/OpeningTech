
import sys, random

if len(sys.argv) > 1:
    random.seed()
    learnfile = "l"+sys.argv[1]
    testfile = "t"+sys.argv[1]
    l= open(learnfile, 'w')
    t= open(testfile, 'w')
    for line in open(sys.argv[1]):
        if (random.random() < 0.1):
            t.write(line)
        else:
            l.write(line)
    l.close()
    t.close()
    print "wrote: ", learnfile, testfile
    sys.exit(0)

random.seed()
l= open("lPvP.txt", 'w')
t= open("tPvP.txt", 'w')
for line in open("PvP.txt"):
    if (random.random() < 0.1):
        t.write(line)
    else:
        l.write(line)
l.close()
t.close()

random.seed()
l = open("lPvT.txt", 'w')
t = open("tPvT.txt", 'w')
for line in open("PvT.txt"):
    if (random.random() < 0.1):
        t.write(line)
    else:
        l.write(line)
l.close()
t.close()

random.seed()
l = open("lPvZ.txt", 'w')
t = open("tPvZ.txt", 'w')
for line in open("PvZ.txt"):
    if (random.random() < 0.1):
        t.write(line)
    else:
        l.write(line)
l.close()
t.close()

random.seed()
l = open("lTvP.txt", 'w')
t = open("tTvP.txt", 'w')
for line in open("TvP.txt"):
    if (random.random() < 0.1):
        t.write(line)
    else:
        l.write(line)
l.close()
t.close()

random.seed()
l = open("lTvT.txt", 'w')
t = open("tTvT.txt", 'w')
for line in open("TvT.txt"):
    if (random.random() < 0.1):
        t.write(line)
    else:
        l.write(line)
l.close()
t.close()

random.seed()
l = open("lTvZ.txt", 'w')
t = open("tTvZ.txt", 'w')
for line in open("TvZ.txt"):
    if (random.random() < 0.1):
        t.write(line)
    else:
        l.write(line)
l.close()
t.close()

random.seed()
l = open("lZvP.txt", 'w')
t = open("tZvP.txt", 'w')
for line in open("ZvP.txt"):
    if (random.random() < 0.1):
        t.write(line)
    else:
        l.write(line)
l.close()
t.close()

random.seed()
l = open("lZvT.txt", 'w')
t = open("tZvT.txt", 'w')
for line in open("ZvT.txt"):
    if (random.random() < 0.1):
        t.write(line)
    else:
        l.write(line)
l.close()
t.close()

random.seed()
l = open("lZvZ.txt", 'w')
t = open("tZvZ.txt", 'w')
for line in open("ZvZ.txt"):
    if (random.random() < 0.1):
        t.write(line)
    else:
        l.write(line)
l.close()
t.close()
