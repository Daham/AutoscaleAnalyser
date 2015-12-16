plotLajCPU = function(config, userCount, t=1, start=-1, end=-1) {
  plotCPU(paste("lajan/", config, "/", userCount, "/cpu", sep=""), t, start, end)
}

plotOurCPU = function(patternName, t=1, start=-1, end=-1) {
  plotCPU(paste("rubis/", patternName, "/cpu", sep=""), t, start, end)
}

plotLajMem = function(config, userCount, t=1, start=-1, end=-1) {
  plotMem(paste("lajan/", config, "/", userCount, "/mem", sep=""), t, start, end)
}

plotOurMem = function(patternName, t=1, start=-1, end=-1) {
  plotMem(paste("rubis/", patternName, "/mem", sep=""), t, start, end)
}

plotRIFMora = function(experiment, t, start=-1, end=-1) {
  plotRIF(paste("moraspirit/", experiment, "/rif", sep=""), t, start, end)
}


plotRIF = function(file, t, start=-1, end=-1) {
  rif = read.csv(file, header=FALSE, stringsAsFactors=FALSE)
  colnames(rif) = c("TS", "total", "value", "load")
  rif = rif[rif$value > -1,]  #remove any error records
  
  if(start > 0 & end > 0) {
    rif = rif[start:end,]
  }
  plotData(rif, t, "Requests in flight")
  
  #proxy$TS = as.POSIXct(as.numeric(proxy$V1)/1000, origin="1970-01-01")
  #plot.ts(proxy$TS, proxy$V3, type="l")
  #x11()
  #plot.ts(row.names(proxy), proxy$V3, type="l")
}


plotCPU = function(file, t, start, end) {
  cpu = read.csv(file, skip=1, sep="", stringsAsFactors=FALSE)
  
  cpu = cpu[cpu[1] != "Average:",]  # remove unnecessary rows & fields
  if("AM" %in% colnames(cpu))
    cpu$AM = NULL
  if("PM" %in% colnames(cpu))
    cpu$PM = NULL
  cpu$CPU = NULL
  
  colnames(cpu) = c("TIME", "USER", "NICE", "SYS", "IO", "STEAL", "IDLE")
  cpu$value = 100 - cpu$IDLE
  if(start > 0 & end > 0) {
    cpu = cpu[start:end,]
  }
  plotData(cpu, t, "CPU usage (%)")
}


plotMem = function(file, t, start, end) {
  mem = read.csv(file, skip=1, sep="", stringsAsFactors=FALSE)
  
  mem = mem[mem[1] != "Average:",]  # remove unnecessary rows & fields
  if("AM" %in% colnames(mem))
    mem$AM = NULL
  if("PM" %in% colnames(mem))
    mem$PM = NULL
  
  colnames(mem) = c("TIME", "KB_FREE", "KB_USED", "value", "BUF", "KB_CACHE", "KB_SWAP_FREE", "KB_SWAP_USED", "SWAP_USED", "KB_SWAP_CAD")
  if(start > 0 & end > 0) {
    mem = mem[start:end,]
  }
  plotData(mem, t, "Memory usage (%)")
}


plotData = function(dataset, t, yLabel) {
  dataset$u = 0
  dataset$u[-(1:t)] = diff(dataset$value, lag=t)/t  #dataset$u[1]: insufficient data
  
  dataset$a = 0
  dataset$a[-(1:t)] = diff(dataset$u, lag=t)/t  #dataset$a[1:2]: insufficient data; dataset$a[2] automatically becomes NA
  
  dataset$s = 0
  dataset$s[-(1:t)] = dataset$value + dataset$u*t + 0.5*dataset$a*t*t		#calculates (u,a)[t-1] -> delta(s[t]); delta(t) = 1
  
  plot(dataset$s, col="#00FF00FF", type="l", xlab="time elapsed (s)", ylab=yLabel)
  lines(dataset$value, type="l", col="#FF0000FF")
}


cat("
    Use setwd() to switch to data directory root before invoking functions.
    
    Usage:
    
    plotLajCPU(machine_config, users_count, time_window, start_record, end_record): data in lajan/
    e.g. plotLajCPU(\"4-5\", 2000, 3, 1, 100) -> records 1-100 from file lajan/4-5/2000/cpu
    
    plotOurCPU(pattern_name, time_window, start_record, end_record): data in rubis/
    e.g. plotOurCPU(\"exponential10\", 3, 1, 100) -> records 1-100 from file rubis/exponential10/cpu

(plotLajMem/plotOurMem used in same fashion)

plotRIFMora(test_name, time_window, start_record, end_record): data in moraspirit/
e.g. plotRIFMora(3, 3, 1, 100) -> records 1-100 from file moraspirit/3/rif

plotCPU(cpu_file, time_window, start_record, end_record)
e.g. plotCPU(\"lajan/4-5/2000/cpu\", 3, 1, 100) -> records 1-100 from file lajan/4-5/2000/cpu

plotMem(mem_file, time_window, start_record, end_record)
e.g. plotMem(\"lajan/4-5/2000/mem\", 3, 1, 100) -> records 1-100 from file lajan/4-5/2000/mem

plotRIF(rif_file, time_window, start_record, end_record)
e.g. plotRIF(\"moraspirit/1/rif\", 3, 1, 100) -> records 1-100 from file moraspirit/1/rif

")