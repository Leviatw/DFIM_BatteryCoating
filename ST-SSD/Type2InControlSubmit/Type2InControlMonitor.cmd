universe = vanilla
getenv = true
executable = /home/tgong33/ImageMonitoring/Type2InControlSubmit/Type2InControlMonitor.sh
arguments = $(ProcId)
output = $ENV(HOME)/ImageMonitoring/InCtrl/Type2/out/job_$(ClusterId)_$(ProcId).out 
error = $ENV(HOME)/ImageMonitoring/InCtrl/Type2/tmp/job_$(ClusterId)_$(ProcId).err
log = $ENV(HOME)/ImageMonitoring/InCtrl/Type2/tmp/job_$(ClusterId).log
queue 4
