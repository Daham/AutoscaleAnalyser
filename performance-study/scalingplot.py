#!/usr/bin/python

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from itertools import izip

# latest datasets
s = {'la': 0, 'rif': 0, 'mc': 0}
u = {'la': 0, 'rif': 0, 'mc': 0}
a = {'la': 0, 'rif': 0, 'mc': 0}

# last prediction for each metric
last_pred = {'la': 0, 'rif': 0, 'mc': 0}

# y-axis plot points
actual = {'la': [], 'rif': [], 'mc': []}
pred = {'la': [], 'rif': [], 'mc': []}
# t-(x-)axis plot points
t_actual = {'la': [], 'rif': [], 'mc': []}
t_pred = {'la': [], 'rif': [], 'mc': []}
# scale up/down times under each metric
t_scaleups = {'la': [], 'rif': [], 'mc': []}
t_scaledowns = {'la': [], 'rif': [], 'mc': []}

# last known instance count
inst_count = 0

# resource stats (calculated)
utilization = {'la': [], 'rif': [], 'mc': []}
allocation = {'la': [], 'rif': [], 'mc': []}
t_utilization = {'la': [], 'rif': [], 'mc': []}
t_allocation = {'la': [], 'rif': [], 'mc': []}

t = 1
ts_init = 0
TIMESCALE = 60	# 60 => show in minutes
FUTURE = 60/TIMESCALE	# prediction interval (s)

# as percentages that can be handled by each machine
HEALTH_SCALE_FACTOR = 100
RIF_SCALE_FACTOR = 100		# TODO need a reasonable value!

SCALE_LOG_REL_PATH = "/repository/logs/scale.log"
DEFAULT_CLUSTER_FILTER = "php"

# parse timestamp
def getTS(timestr):
	return time.mktime(time.strptime(timestr, '[%Y-%m-%d %H:%M:%S,%f]'))/TIMESCALE

# record resource utilization and instance up/down events
def recordInstances(ts, scaleList=None):
	for metric in ['la', 'rif', 'mc']:
		allocation[metric].append(inst_count)
		utilization[metric].append(s[metric]*inst_count/(RIF_SCALE_FACTOR if metric == 'rif' else HEALTH_SCALE_FACTOR))
		t_allocation[metric].append(ts)
		t_utilization[metric].append(ts)
		if scaleList is not None and line.find(metric) > 0:
			scaleCount = len(scaleList[metric])
			# add timestamp, avoiding overlapping requests
			scaleList[metric].append(ts + (0.3 if scaleCount > 0 and scaleList[metric][scaleCount - 1] == ts else 0))

print "\nUsage:\tpython mock.py log_file_path cluster_filter"
print "\tDefault log_file_path:\t$CARBON_HOME" + SCALE_LOG_REL_PATH
print "\tDefault cluster_filter:\t*" + DEFAULT_CLUSTER_FILTER + "*\n"

# fall back to default cluster ID if not provided
clusterId = DEFAULT_CLUSTER_FILTER if len(sys.argv) < 3 else sys.argv[2]

f = open(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CARBON_HOME") + SCALE_LOG_REL_PATH, "r")
for line in f:
	try:
		# irrelevant?
		if line.find('HealthStat') < 0 or line.find(clusterId) < 0:	# ignore MySQL stats
			continue

		tok = line.split(' ')
		ts = getTS(tok[0] + ' ' + tok[1])

		# start time
		if ts_init == 0:
			ts_init = ts
		ts -= ts_init

		if tok[2] == 'HealthStatEvent':
			pass	# not used currently

		elif tok[2] == 'HealthStatProc':
			if tok[3] == 'la' or tok[3] == 'mc' or tok[3] == 'rif':
				metric = tok[3]
			
				avg = float(tok[6])
				actual[metric].append(avg)
				t_actual[metric].append(ts)
			
				# prediction
				if len(tok) > 9:
					last_pred[metric] = float(tok[9])
				else:
					last_pred[metric] = s[metric] + u[metric]*t + 0.5*a[metric]*t*t
				pred[metric].append(last_pred[metric])
				t_pred[metric].append(ts + FUTURE)

				# latest entries, for next prediction
				s[metric] = avg
				u[metric] = float(tok[7])
				a[metric] = float(tok[8])
				#print("%d %9f %9f %9f" % (ts, s[metric], u[metric], a[metric]))

			elif tok[3] == 'count':	# just an instance count update
				inst_count = int(tok[6])
				recordInstances(ts)

			elif tok[3] == '+':	# scale-up event
				#inst_count += 1
				recordInstances(ts, t_scaleups)

			elif tok[3] == '-':	# scale-down event
				#inst_count -= 1
				recordInstances(ts, t_scaledowns)

	except BaseException as e:
		print(e)
		print(line)
		break

# write results to files
for metric in ['la', 'rif', 'mc']:
	out = open(metric, "w")

	stats = actual[metric]
	t_stats = t_actual[metric]
	counts = allocation[metric]
	t_counts = t_allocation[metric]

	k = 0
	for i, t in izip(stats, t_stats):
		while t_counts[k] < t:
			k += 1
		out.write(str(i) + "," + str(counts[k]) + "\n")
	out.close()

# plot separate graphs
for metric in ['la', 'rif', 'mc']:
	plt.figure()
	
	# actual and predicted
	#plt.plot(np.array(t_actual[metric]), np.array(actual[metric]))
	#plt.plot(np.array(t_pred[metric]), np.array(pred[metric]))
	
	plt.step(np.array(t_allocation[metric]), np.array(allocation[metric]), color="blue", label="allocation")
	plt.plot(np.array(t_utilization[metric]), np.array(utilization[metric]), color="black", label="utilization")
	
	# scale up/down timestamps
	scaleLabeled = False
	for t in t_scaleups[metric]:
		plt.axvline(t, ls=":", color="red", label=None if scaleLabeled else "scale up")
		scaleLabeled = True	# add label only to first line

	scaleLabeled = False
	for t in t_scaledowns[metric]:
		plt.axvline(t, ls=":", color="green", label=None if scaleLabeled else "scale down")
		scaleLabeled = True

	plt.title(metric)
	plt.xlabel("time | " + str(TIMESCALE) + "s")
	plt.ylabel("machine units")
	plt.legend(loc="best")

plt.show()
