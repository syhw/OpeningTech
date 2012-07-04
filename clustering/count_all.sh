#!/bin/bash
unknown=0
two_gates=0
fast_dt=0
templar=0
speedzeal=0
corsair=0
nony=0
reaver_drop=0
for name in `ls my*Protoss*`
do 
    echo $name
    echo 'unknown:' && cat $name | grep 'unknown' | wc -l
    ((unknown = unknown+`cat $name | grep 'unknown' | wc -l`))
    echo 'two_gates:' && cat $name | grep 'two_gates' | wc -l
    ((two_gates = two_gates+`cat $name | grep 'two_gates' | wc -l`))
    echo 'fast_dt:' && cat $name | grep 'fast_dt' | wc -l
    ((fast_dt = fast_dt+`cat $name | grep 'fast_dt' | wc -l`))
    echo 'templar:' && cat $name | grep 'templar' | wc -l
    ((templar = templar+`cat $name | grep 'templar' | wc -l`))
    echo 'speedzeal:' && cat $name | grep 'speedzeal' | wc -l
    ((speedzeal = speedzeal+`cat $name | grep 'speedzeal' | wc -l`))
    echo 'corsair:' && cat $name | grep 'corsair' | wc -l
    ((corsair = corsair+`cat $name | grep 'corsair' | wc -l`))
    echo 'nony:' && cat $name | grep 'nony' | wc -l
    ((nony = nony+`cat $name | grep 'nony' | wc -l`))
    echo 'reaver_drop:' && cat $name | grep 'reaver_drop' | wc -l
    ((reaver_drop = reaver_drop+`cat $name | grep 'reaver_drop' | wc -l`))
done
total=0
((total=unknown+two_gates+fast_dt+templar+speedzeal+corsair+nony+reaver_drop))
echo 'Protoss:'
echo 'two_gates:' && echo $two_gates
echo 'fast_dt:' && echo $fast_dt
echo 'templar:' && echo $templar
echo 'speedzeal:' && echo $speedzeal
echo 'corsair:' && echo $corsair
echo 'nony:' && echo $nony
echo 'reaver_drop:' && echo $reaver_drop
echo 'unknown:' && echo $unknown
echo 'total:' && echo $total

for name in `ls my*Terran*`
do 
    echo $name
    echo 'unknown:' && cat $name | grep 'unknown' | wc -l
    echo 'bio:' && cat $name | grep 'bio' | wc -l
    echo 'rax_fe:' && cat $name | grep 'rax_fe' | wc -l
    echo 'two_facto:' && cat $name | grep 'two_facto' | wc -l
    echo 'vultures:' && cat $name | grep 'vultures' | wc -l
    echo 'drop:' && cat $name | grep 'drop' | wc -l
done
for name in `ls my*Zerg*`
do 
    echo $name
    echo 'unknown:' && cat $name | grep 'unknown' | wc -l
    echo 'speedlings:' && cat $name | grep 'speedlings' | wc -l
    echo 'fast_mutas:' && cat $name | grep 'fast_mutas' | wc -l
    echo 'mutas:' && cat $name | grep 'mutas' | wc -l
    echo 'lurkers:' && cat $name | grep 'lurkers' | wc -l
    echo 'hydras:' && cat $name | grep 'hydras' | wc -l
done
#!/bin/bash
for name in `ls scm*Protoss*`
do 
    echo $name
    echo 'FastLegs:' && cat $name | grep 'FastLegs' | wc -l
    echo 'FastDT:' && cat $name | grep 'FastDT' | wc -l
    echo 'FastObs:' && cat $name | grep 'FastObs' | wc -l
    echo 'ReaverDrop:' && cat $name | grep 'ReaverDrop' | wc -l
    echo 'Carrier:' && cat $name | grep 'Carrier' | wc -l
    echo 'FastExpand:' && cat $name | grep 'FastExpand' | wc -l
    echo 'Unknown:' && cat $name | grep 'Unknown' | wc -l
done
for name in `ls scm*Terran*`
do 
    echo $name
    echo 'Bio:' && cat $name | grep 'Bio' | wc -l
    echo 'TwoFactory:' && cat $name | grep 'TwoFactory' | wc -l
    echo 'VultureHarass:' && cat $name | grep 'VultureHarass' | wc -l
    echo 'SiegeExpand:' && cat $name | grep 'SiegeExpand' | wc -l
    echo 'Standard:' && cat $name | grep 'Standard' | wc -l
    echo 'FastDropship:' && cat $name | grep 'FastDropship' | wc -l
    echo 'Unknown:' && cat $name | grep 'Unknown' | wc -l
done
for name in `ls scm*Zerg*`
do 
    echo $name
    echo 'HatchMuta:' && cat $name | grep 'HatchMuta' | wc -l
    echo 'ThreeHatchMuta:' && cat $name | grep 'ThreeHatchMuta' | wc -l
    echo 'TwoHatchMuta:' && cat $name | grep 'TwoHatchMuta' | wc -l
    echo 'HydraRush:' && cat $name | grep 'HydraRush' | wc -l
    echo 'Standard:' && cat $name | grep 'Standard' | wc -l
    echo 'HydraMass:' && cat $name | grep 'HydraMass' | wc -l
    echo 'Unknown:' && cat $name | grep 'Unknown' | wc -l
    echo 'Lurker:' && cat $name | grep 'Lurker' | wc -l
done
