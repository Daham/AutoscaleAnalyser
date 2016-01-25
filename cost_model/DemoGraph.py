__author__ = 'bhash90'
import matplotlib.pyplot as plt
import numpy

def plot_demo() :
    x = numpy.linspace(0,25,100) # 100 linearly spaced numbers
    y = -(x-2)*(x-22)
    # compose plot
    f, (pylab) = plt.subplots(1,1, sharex = True)
    pylab.plot(x,y, c= "black" , linewidth = 1)
    pylab.axhline(y=20,xmin=0,xmax=25,linewidth=2.5,zorder=0)
    pylab.axhline(y=40,xmin=0,xmax=25,linewidth=1.25,zorder=0)
    pylab.axhline(y=80,xmin=0,xmax=25,linewidth=0.7,zorder=0)
    pylab.axhline(y=120,xmin=0,xmax=25,linewidth=0.25,zorder=0)
    pylab.set_xticklabels([])
    pylab.set_yticklabels([])
    pylab.set_xlabel("Time/minutes")
    pylab.set_ylabel("Number of VMs")
    pylab.legend(["Predicted Workload", "VM_Count = Min", "VM Count = Low", "VM Count = Mid", "VM Count = Max" ], loc='upper left', prop={'size':8})
    plt.show()
plot_demo()