#!/usr/bin/python

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from itertools import izip

PRICE = 1
SCALE_LOG_REL_PATH = "/repository/logs/scale.log"
DEFAULT_CLUSTER_FILTER = "php"
TIMESCALE = 60	# 60 => show in minutes
ts_init = 0

# parse timestamp
def getTS(timestr):
	return time.mktime(time.strptime(timestr, '[%Y-%m-%d %H:%M:%S,%f]'))/TIMESCALE

# fall back to default cluster ID if not provided
clusterId = DEFAULT_CLUSTER_FILTER if len(sys.argv) < 4 else sys.argv[3]

instances = []
t_instances = []
i = 0
labels = ["high threshold", "low threshold"]

j = 0
for name in [sys.argv[1], sys.argv[2]]:
	instances.append([PRICE])
	t_instances.append([0])
	i = 0
	ts_init = 0
	
	f = open(name, "r")
	for line in f:
		try:
			# irrelevant?
			if line.find('HealthStat') < 0 or line.find(clusterId) < 0:	# ignore other logs/clusters
				continue

			tok = line.split(' ')
			ts = getTS(tok[0] + ' ' + tok[1])

			# start time
			if ts_init == 0:
				ts_init = ts
			ts -= ts_init

			if tok[2] == 'HealthStatProc':
				if tok[3] == '+':	# scale-up event
					i += 1
					instances[j].append(i*PRICE)
					t_instances[j].append(ts)

		except BaseException as e:
			print(e)
			print(line)
			break

	instances[j].append(i*PRICE)
	t_instances[j].append(ts)
	j += 1

# reduce data from second set
for i in range(0, len(t_instances[1])-1):
	t_instances[1][i] -= 10.52

# plot
plt.step(np.array(t_instances[0]), np.array(instances[0]), label=labels[0])
plt.step(np.array(t_instances[1]), np.array(instances[1]), label=labels[1])
plt.xlabel("time | minutes")
plt.ylabel("machine cost units")
plt.legend(loc="best")
plt.show()
