#!/usr/bin/python

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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

# parse timestamp
def getTS(timestr):
	return time.mktime(time.strptime(timestr, '[%Y-%m-%d %H:%M:%S,%f]'))/TIMESCALE

# record resource utilization and instance up/down events
def recordInstances(ts, scaleList=None):
	for metric in ['la', 'rif', 'mc']:
		allocation[metric].append(inst_count)
		utilization[metric].append(s[metric]*inst_count/(1 if metric == 'rif' else 100))
		t_allocation[metric].append(ts)
		t_utilization[metric].append(ts)
		if scaleList is not None and line.find(metric):
			scaleList[metric].append(ts)

f = open(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CARBON_HOME") + "/repository/logs/scale.log", "r")
for line in f:
	try:
		# irrelevant?
		if line.find('HealthStat') < 0:
			continue

		tok = line.split(' ')
		ts = getTS(tok[0] + ' ' + tok[1]) - ts_init
		# start time
		if ts_init == 0:
			ts_init = ts

		if tok[2] == 'HealthStatEvent':
			pass	# not used currently

		elif tok[2] == 'HealthStatProc':
			if tok[3] == 'la' or tok[3] == 'mc' or tok[3] == 'rif':
				metric = tok[3]
			
				avg = float(tok[6])
				actual[metric].append(avg)
				t_actual[metric].append(ts)
			
				# prediction
				t_pred[metric].append(ts + FUTURE)
				last_pred[metric] = s[metric] + u[metric]*t + 0.5*a[metric]*t*t
				pred[metric].append(last_pred[metric])

				# latest entries, for next prediction
				s[metric] = avg
				u[metric] = float(tok[7])
				a[metric] = float(tok[8])
				#print("%d %9f %9f %9f" % (ts, s[metric], u[metric], a[metric]))

			elif tok[3] == 'count':	# just an instance count update
				inst_count = int(tok[6])
				recordInstances(ts)

			elif tok[3] == '+':	# scale-up event
				inst_count += 1
				recordInstances(ts, t_scaleups)

			elif tok[3] == '-':	# scale-down event
				inst_count -= 1
				recordInstances(ts, t_scaledowns)

	except BaseException as e:
		print(e)
		print(line)
		break

# plot separate graphs
for metric in ['la', 'rif', 'mc']:
	plt.figure()
	
	# actual and predicted
	#plt.plot(np.array(t_actual[metric]), np.array(actual[metric]))
	#plt.plot(np.array(t_pred[metric]), np.array(pred[metric]))
	
	# VM allocation and actual utilization
	plt.step(np.array(t_allocation[metric]), np.array(allocation[metric]), color="blue")
	plt.plot(np.array(t_utilization[metric]), np.array(utilization[metric]), "*", color="black")

	plt.title(metric)
	plt.xlabel("Time | s")
	plt.ylabel(metric)
	plt.legend(["Allocation", "Utilization"], loc='upper right')
	
	# scale up/down timestamps
	for t in t_scaleups[metric]:
		plt.axvline(t, color="red")
	for t in t_scaledowns[metric]:
		plt.axvline(t, color="green")
	
	# re-plot allocation to overlay scaling lines
	plt.step(np.array(t_allocation[metric]), np.array(allocation[metric]), color="blue")

plt.show()
