#!/bin/bash
source ./gurobi-env/bin/activate
python /home/tgong33/ImageMonitoring/Type1InControlSubmit/Type1InControlMonitor.py -- $1
deactivate