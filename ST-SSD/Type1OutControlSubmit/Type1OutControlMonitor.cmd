universe = vanilla
getenv = true
executable = /home/tgong33/ImageMonitoring/Type1OutControlSubmit/Type1OutControlMonitor.sh
arguments = $(ProcId)
output = $ENV(HOME)/ImageMonitoring/OutCtrl/Type1/out/job_$(ClusterId)_$(ProcId).out 
error = $ENV(HOME)/ImageMonitoring/OutCtrl/Type1/tmp/job_$(ClusterId)_$(ProcId).err
log = $ENV(HOME)/ImageMonitoring/OutCtrl/Type1/tmp/job_$(ClusterId).log
queue 24
