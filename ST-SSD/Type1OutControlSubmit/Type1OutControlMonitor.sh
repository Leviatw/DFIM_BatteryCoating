#!/bin/bash
source ./gurobi-env/bin/activate
python /home/tgong33/ImageMonitoring/Type1OutControlSubmit/Type1OutControlMonitor.py -- $1
deactivate