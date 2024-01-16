#!/bin/bash
source ./gurobi-env/bin/activate
python /home/tgong33/ImageMonitoring/Type2OutControlSubmit/Type2OutControlMonitor.py -- $1
deactivate