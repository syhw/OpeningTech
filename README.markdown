# Opening prediction for StarCraft: Broodwar

Dependency:  
[ProBT](http://probayes.com/index.php?option=com_content&view=article&id=83&Itemid=88&lang=en)

Input format:  
One replay/game per line, lines such as  
    First_Building_Name Time_Built; Second_Building_Name Time_Built2;

It does:  
- learn the possible tech trees from replays (or you can generate them)
- learn the distribution of P(Time | X, Opening)
- infer P(Opening | Observations, Time)

