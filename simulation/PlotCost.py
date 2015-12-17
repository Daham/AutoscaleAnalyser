__author__ = 'ridwan'


import csv
import array
import matplotlib.pyplot as plt
import numpy as np

#normal cost and optimized cost details, workload details provided by CSV
def plot(workload, normalCost, optimizedCost, workloadType):

    workloadTypes = ["CPU Usage", "Memory Consumption","Request in Flight Count"]
    f, (workloadGraph, costGraph) = plt.subplots(1, 2, sharex=True)
    f.suptitle("Normal cost vs Optimized cost")

    plotWorkload(workload,workloadGraph,workloadTypes[workloadType])

    costGraph.set_xlabel("Time (minutes)")
    costGraph.set_ylabel("Total Cost")
    costGraph.set_title("Cumulative Cost")

    plotCost(normalCost,costGraph)
    plotCost(optimizedCost,costGraph)
    costGraph.legend(["Reactive", "Proactive" ], loc='upper left')

    plt.show()

def plotCost(filename,plotArea):

    x_coordinates = array.array('d')
    y_coordinates = array.array('d')

    fileData = open(filename,"rb")
    reader = csv.reader(fileData)

    rownum = 0
    for row in reader:
        if rownum == 0:
            rownum += 1
        else:
            x_coordinates.append(float(row[0]))
            y_coordinates.append(float(row[1]))
        rownum += 1
    fileData.close()

    # Regression
    xdata = np.array(x_coordinates)
    ydata = np.array(y_coordinates)

    #Plot row data
    plotArea.plot(xdata, ydata)


def plotWorkload(filename, plotArea,workloadType):

    x_coordinates = array.array('d')
    y_coordinates = array.array('d')

    fileData = open(filename,"rb")
    reader = csv.reader(fileData)

    rownum = 0
    for row in reader:
        if rownum == 0:
            rownum += 1
        else:
            x_coordinates.append(float(row[0]))
            y_coordinates.append(float(row[1]))
        rownum += 1
    fileData.close()

    # Regression
    xdata = np.array(x_coordinates)
    ydata = np.array(y_coordinates)

    #Plot row data
    plotData = plotArea.plot(xdata, ydata)
    plotArea.set_xlabel("Time (minutes)")
    plotArea.set_ylabel(workloadType)
    plotArea.set_title("Workload")
    #plotArea.setp(lines, color='r', linewidth=2.0)


# 0 - CPU, 1 - Memory, 2 - RIF
plot("data/rif_exp_wl.csv","data/normal_cost.csv","data/optimized_cost.csv",2)




