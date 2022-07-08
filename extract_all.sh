#!/bin/bash
#
# Script for spawning a different python process
# for linkgraph extraction for each year.
# The processes will run in the background with the help of nohup
# and log to their separate extraction_<year>.out file.

for year in {2006..2016}
do
    nohup python3 -u extract_linkgraph.py "${year}" &> "extraction_${year}.out" &
done