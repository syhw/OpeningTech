#!/opt/local/bin/python2.6

# http://www.teamliquid.net/replay/
# http://www.gosugamers.net/starcraft/replays/
import urllib2

#TEST showing that the redirect works
#url = 'http://www.teamliquid.net/replay/download.php?replay=1801'
#url = 'http://www.gosugamers.net/starcraft/admin/a_replays.php?dl=30084'
#page = urllib2.urlopen(url)
#print page.geturl()

url = 'http://www.teamliquid.net/replay/download.php?replay='
for i in range(1964):
    try:
        page = urllib2.urlopen(url+str(i+1))
    except: 
        continue
    repnamel = page.geturl().split('/')
    repname = repnamel[len(repnamel)-1]
    file = open('teamliquid/'+repname,"wb")
    file.write(page.read())
    print "saved replay ",
    print repname
    file.close()



