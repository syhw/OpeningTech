# Opening prediction for StarCraft: Broodwar

Dependency:  
[ProBT](http://probayes.com/index.php?option=com_content&view=article&id=83&Itemid=88&lang=en)

Input format:  
One replay/game per line, lines such as  
    First_Building_Name Time_Built; Second_Building_Name Time_Built2;

It does:  

1. learn the possible tech trees (X) from replays (or you can generate them)
2. learn the distributions of P(Time | X, Opening) and P(X | Opening)
3. infer P(Opening | Observations, Time)

See *model.pdf*.
