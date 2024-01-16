universe = vanilla
getenv = true
executable = /home/tgong33/ImageMonitoring/Type1InControlSubmit/Type1InControlMonitor.sh
arguments = $(ProcId)
output = $ENV(HOME)/ImageMonitoring/InCtrl/Type1/out/job_$(ClusterId)_$(ProcId).out 
error = $ENV(HOME)/ImageMonitoring/InCtrl/Type1/tmp/job_$(ClusterId)_$(ProcId).err
log = $ENV(HOME)/ImageMonitoring/InCtrl/Type1/tmp/job_$(ClusterId).log
queue 4
