#!/bin/bash
source ./gurobi-env/bin/activate
python /home/tgong33/ImageMonitoring/Type2InControlSubmit/Type2InControlMonitor.py -- $1
deactivate