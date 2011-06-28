#!/bin/bash
for ((i = 0; i <= 45648; i++));
do
    wget http://www.gosugamers.net/starcraft/admin/a_replays.php?dl=$i
done
rm index.html*
