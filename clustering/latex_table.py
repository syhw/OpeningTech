f = open('TEMP.txt', 'r')

mu = ""
a = {}
previous = ""
for line in f:
    #print previous
    l = line.rstrip('\r\n')
    if "scm" in l:
        active = False
    if "myscm" in l:
        active = True
        first = l.split('_')[1][0]
        fakemu = l.split('_')[0][-3:]
        if first == fakemu[0]:
            mu = fakemu
        else:
            mu = first + 'v' + fakemu[0]
        a[mu] = {}
    if active:
        if ':' in l:
            l = l.rstrip(':')
            #print l
            previous = l
        else:
            a[mu][previous] = l.strip(' ')

print a

for KK in range(10):
    i = 0
    s = "" 
    for mu in ['PvP','PvT','PvZ','TvP','TvT','TvZ','ZvP','ZvT','ZvZ']:
        try:
            op, nb = a[mu].popitem()
        except:
            op = ""
            nb = ""
        if (i % 3) == 0:
            s += op + " & "
        s += nb + " & "
        i += 1
    print s





