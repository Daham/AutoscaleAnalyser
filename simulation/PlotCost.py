__author__ = 'ridwan'


import csv
import array
import matplotlib.pyplot as plt
import numpy as np


def plot(normalCost, optimizedCost):

    f, (plt1, plt2) = plt.subplots(1, 2, sharex=True)
    f.suptitle("Normal cost vs Optimized cost")

    plotGraph(normalCost,plt1,"Normal Cost")
    plotGraph(optimizedCost,plt2,"Optimized Cost")
    plt.show()


def plotGraph(filename, plotArea,graphName):

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
    plotArea.set_ylabel("Total cost")
    plotArea.set_title(graphName)
    #plotArea.setp(lines, color='r', linewidth=2.0)
    #plotArea.legend(["Raw Data", "Predicted", "Blind Killing","Smart Killing" ], loc='upper right')


plot("data/cpu_exp_wl.csv","data/cpu_exp_wl.csv")




