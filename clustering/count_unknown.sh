#!/bin/bash
for name in `ls my*`; do echo $name && cat $name | grep 'unknown' | wc -l; done
