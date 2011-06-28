# Opening/BuildTree prediction for StarCraft: Broodwar

### Dependency:  
[ProBT](http://probayes.com/index.php?option=com_content&view=article&id=83&Itemid=88&lang=en)  
[Boost.serialization](http://www.boost.org/doc/libs/1_45_0/libs/serialization/doc/index.html) (optional, note: compiled with `./bjam --with-serialization --build-type=complete --layout=versioned toolset=darwin architecture=x86 address-model=32`)

### Input format:  
One replay/game per line, lines such as  
    First_Building_Name Time_Built; Second_Building_Name Time_Built2;

See the \*v\*.txt files

### It does:  

1. Learn the possible tech trees (X) from replays (or you can generate them)
2. Learn the distributions of P(Time | X, Opening) and P(X | Opening)
3. Learn the distributions of P(Time | BuildTree) with TECH\_TREES switched on
4. Infer P(Opening | Observations, Time)

See *model.pdf* or [CIG'11 publication](http://dl.dropbox.com/u/14035465/OpeningPrediction.pdf) for 1,2,4 and [AIIDE'11 publication](http://dl.dropbox.com/u/14035465/AIIDE_11_RC1.pdf) for 3.

### Workflow:

1. (Optional, our labels) Clusterize from clustering/launch\_labeling.sh
2. Transform (transmute) arff files into txt files with scripts/transmute.sh
3. To perform evaluations, cut the txt files into learn and test files with the scripts/cut\* scripts
4. Compile (make) and test either with make targets or `export (DY)LD_LIBRARY_PATH=<probt/lib(+boost/lib)> && ./model lXvX.txt tXvX.txt`


