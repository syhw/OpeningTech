echo ""
echo "BW labels, noise $1:"
for name in [TPZ]v[TPZ].txt
do
    echo ${name%.*}
    ./model l$name t${name%.*}n$1.txt | grep ">>> Positive classif"
done
echo ""
echo "MY labels, noise $1"
for name in [TPZ]v[TPZ]x.txt
do
    echo ${name%.*}
    ./mymodel l$name t${name%.*}n$1.txt | grep ">>> Positive classif"
done
