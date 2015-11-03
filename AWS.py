from distutils.command.upload import upload

__author__ = 'bhash90'
#Todo Graphs should be in same figure where necessary
#Todo Name Axes and lines
import math
import csv
import array
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import EMA

def quad(x, a, b, c, d):
    return a * x*x*x*x*x + b*x*x*x*x + c*x*x*x + d

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def drange(x, y, jump):
    while x < y:
        yield x
        x += jump

class VM:
    VM_id = 0
    expected = 0		#min. expected # of VMs to be kept alive
    def __init__(self, initTime, endTime):
        VM.VM_id += 1
        self.initTime = initTime
        self.endTime = endTime
        self.id = VM.VM_id;
        #self.timeStamp = timeStamp # may be required for azure

def startVMs(listVM, count, i):
    for k in range(0, count):
        vm = VM(initTime=i, endTime=-1)
        listVM.append(vm)
    return count

#random remove implemented. Check the provider and implement as a switch
def removeVMs(listVM, count, provider, now):
    AWS_MIN_MINUTES_BEFORE_KILL = 50	#don't kill if less than this # minutes of last hour is spent
    AWS_MAX_MINUTES_BEFORE_KILL = 57	#don't kill if more than this # minutes of last hour is spent
    
    killed = 0
    candidates = []
    
    for vm in listVM:
        if vm.endTime == -1:
            if provider == "default":
                candidates.append((vm, 0))	#dummy value for hourFraction
                if len(candidates) == count:
                    break
            elif provider == "aws":		#first pick all candidates
                hourFraction = (now - vm.initTime)%60
                if AWS_MIN_MINUTES_BEFORE_KILL < hourFraction < AWS_MAX_MINUTES_BEFORE_KILL:
                    candidates.append((vm, hourFraction))

    if provider == "aws":		#sort descending w.r.t. hourFraction
        candidates.sort(lambda x, y: (int)(y[1] - x[1]))

    #kill as many as possible/required
    for (vm, hourFraction) in candidates:
        vm.endTime = now
        #print("End VM %s at %s" % (vm.id, now))
        killed = killed + 1
        if killed == count:
            break
    return killed

def calculateAWSCost(listVM, givenTime, price_per_hour):
    cost = 0
    for vm in listVM:
        if(vm.initTime < givenTime):
            if(vm.endTime == -1 or (vm.endTime > givenTime)):
                billing_hours = int(math.ceil((givenTime - vm.initTime)/60))
                cost += billing_hours * price_per_hour
                #print("cost: %s at %s and %s init %s" % (cost,givenTime,vm.endTime, vm.initTime))
            else:
                billing_hours = int(math.ceil((vm.endTime - vm.initTime)/60))
                cost += billing_hours * price_per_hour
    #print("cost: %s at %s" % (cost,givenTime))
    return cost

def getValue(line2d, x):
    xvalues = line2d[0].get_xdata()
    yvalues = line2d[0].get_ydata()
    idx = np.where(xvalues == xvalues[x])
    return yvalues[idx]

def mse(predictions, targets):
    #print(predictions)
    #print(targets)
    e = np.sqrt(np.mean((predictions - targets)**2))
    #print(e)
    return e

def run(VM_parameter_unit, threshold_percentage, uptime, min_VM, shift, provider, vm_price_per_hour, vm_init_data):
    listVM = list()
    f, (plt1, plt2) = plt.subplots(1, 2, sharex=True)
    print(provider)
    plt.title(provider)
    
    x_coordinates = array.array('d')
    y_coordinates = array.array('d')
    digix_cordinates = array.array('d')
    digiy_cordinates = array.array('d')
    digiy_coord_actual = array.array('d')

    #Read vlaues while converting to VM_UNITS
    ifile = open("rif_exp_wl.csv", "rb")
    reader = csv.reader(ifile)
    rownum = 0
    for row in reader:
        if rownum == 0:
            rownum += 1
        else:
            x_coordinates.append(float(row[0]))
            y_coordinates.append(float(row[1])/(VM_parameter_unit))
        rownum += 1
    ifile.close()
    #print(x_coordinates)
    #print(y_coordinates)

    # Regression
    xdata = np.array(x_coordinates)
    ydata = np.array(y_coordinates)

    #Plot row data
    plt1.plot(xdata, ydata, '*')

    #plot regression line of data
    #plt1.plot(xdata, quad(xdata,popt[0],popt[1],popt[2], popt[3]), '-')
    #plot EMA
    line2d = plt1.plot(xdata, EMA.ema(ydata ,3))

    #Initialize min_VMs
    #VM = [23,42] #Todo fill with randoms

    for j in range(0, min_VM):
        # id = randint(1,999)
        t = vm_init_data[j]#take this as arg?
        vm = VM(initTime=-t, endTime=-1)
        listVM.append(vm)

    #Plot number of VMs required
    vm_count = min_VM
    yvalueset = []
    
    for i in drange(uptime, max(xdata) - shift + uptime, 0.1):
        z = getValue(line2d, i - uptime)
        yvalueset.append(z)
        #print(z)
        new_vm_count = math.ceil(z/threshold_percentage)
        if new_vm_count < min_VM:
            new_vm_count = min_VM

        vm_change = int(math.ceil(new_vm_count - vm_count))
        if vm_change > 0:
            vm_count += startVMs(listVM, vm_change, i)
        elif vm_change < 0:
            vm_count -= removeVMs(listVM, -vm_change, provider, i)
        
        digix_cordinates.append(i)
        digiy_cordinates.append(new_vm_count)
        digiy_coord_actual.append(vm_count)

    digixdata = np.array(digix_cordinates)
    digiydata = np.array(digiy_cordinates)
    plt1.plot(digixdata, digiydata)		#requirement
    plt1.plot(digixdata, np.array(digiy_coord_actual))	#actual

    for vm in listVM:
        print("VM_id %3s: %5.1f - %5.1f = %5.1f%s" % (vm.id, vm.initTime, vm.endTime, 
        (vm.endTime if vm.endTime > 0 else i) - vm.initTime, ("" if vm.endTime > 0 else " up")))

    costValues = array.array('d')
    for k in drange(0, max(xdata), 10):
        costValues.append(calculateAWSCost(listVM, k, vm_price_per_hour))
    costydata = np.array(costValues)
    costxdata = np.arange(0, max(xdata), 10)

    plt2.plot(costxdata,costydata)

    yvalues = np.array(yvalueset)
    e = mse(digiydata, yvalues)
    return e

e = run(16, .8, 10, 2, 0, "aws", 6, [20,40])
e = run(16, .8, 10, 2, 0, "default", 6, [20,40])
plt.show()

