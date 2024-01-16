universe = vanilla
getenv = true
executable = /home/tgong33/ImageMonitoring/Type2OutControlSubmit/Type2OutControlMonitor.sh
arguments = $(ProcId)
output = $ENV(HOME)/ImageMonitoring/OutCtrl/Type2/out/job_$(ClusterId)_$(ProcId).out 
error = $ENV(HOME)/ImageMonitoring/OutCtrl/Type2/tmp/job_$(ClusterId)_$(ProcId).err
log = $ENV(HOME)/ImageMonitoring/OutCtrl/Type2/tmp/job_$(ClusterId).log
queue 40
