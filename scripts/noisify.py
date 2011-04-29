#!/opt/local/bin/python
import sys, random, math

if len(sys.argv) > 1:
    random.seed()
    for i in range(0,5):
        fil = sys.argv[1].split('.')[0]+'n'+str(i)+'.txt'
        f = open(fil, 'w')
        for line in open(sys.argv[1]):
            elems = line.split(';')
            for n in range(i):
                elems.pop(int(math.floor(random.uniform(0, len(elems)-0.01))))
            f.write(';'.join(elems))
        f.close()
        print "wrote: ", f
else:
    print "need one argument file"

