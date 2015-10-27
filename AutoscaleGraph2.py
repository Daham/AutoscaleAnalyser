__author__ = 'bhash90'
#Todo Graphs should be in dame figur where necessary
#Todo Name Aixs and lines
import math
import csv
import array
import matplotlib.pyplot as plt
import numpy as np;
from scipy.optimize import curve_fit
from random import randint
import EMA

def quad(x, a, b, c, d):
     return a * x*x*x*x*x+ b*x*x*x*x + c*x*x*x + d

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def drange(x, y, jump):
  while x < y:
    yield x
    x += jump
class VM:
    VM_id = 0
    def __init__(self, initTime,  endTime):
        VM.VM_id += 1
        self.initTime = initTime
        self.endTime = endTime
        self.id = VM.VM_id;
       # self.timeStamp = timeStamp # may be required for azure

def startVM(listVM, i):
    vm = VM(initTime = i, endTime= -1)
    listVM.append(vm)

#random remove implemented. Check he provider and  implement as a switch
def removeVM(listVM, provider, j):
    #Todo  implement smart killing under removeVM

    AWS_MIN_MINUTES_BEFORE_KILL = 50	#don't kill if less than 45 min of last hour is spent
    AWS_MAX_MINUTES_BEFORE_KILL = 57	#don't kill if more than 55 min of last hour is spent

    if(provider == "default"):
        candidate = None
        for vm in listVM:
            if vm.endTime == -1:
                    candidate = vm
                    break
        if candidate is None:
            pass
        else:
            vm.endTime = j
        print("Default: End VM %s at %s" % (vm.id, j))
    elif provider == "aws":
        maxHourFraction = AWS_MIN_MINUTES_BEFORE_KILL
        candidate = None
        for vm in listVM:
            hourFraction = (j - vm.initTime)%60
            if vm.endTime == -1 and hourFraction > maxHourFraction and hourFraction < AWS_MAX_MINUTES_BEFORE_KILL:
                maxHourFraction = hourFraction
                candidate = vm
                break
        if candidate is None:
            pass		#can't kill any machine
        else:
            vm.endTime = j
            print("AWS:End VM %s at %s" % (vm.id, j))

    return listVM

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
def getVaue(line2d,x):
    xvalues = line2d[0].get_xdata()
    yvalues = line2d[0].get_ydata()
    idx = np.where(xvalues==xvalues[x])
    return yvalues[idx]



def run(VM_parameter_unit, threshold_prcentage,uptime, min_VM,shift,provider,vm_price_per_hour, ):
    listVM = list()
    f, (plt1,plt2) = plt.subplots(1,2,sharex= True)
    VM_PARAMETER_UNIT = VM_parameter_unit #size ofthe parameter in a single node
    VM_THRESHOLD_PRECENTAGE = threshold_prcentage #threshold level defined by user
    VM_UPTIME = uptime #how long does it take to spin up an instance
    MIN_VM = min_VM
    SHIFT = shift
    PROVIDER = provider #read as user arg
    VM_PRICE_PER_HOUR = vm_price_per_hour

    x_coordinates = array.array('d')
    y_coordinates =array.array('d')
    digix_cordinates = array.array('d')
    digiy_cordinates = array.array('d')

    #Read vlaues while converting to VM_UNITS
    ifile  = open('rif.csv', "rb")
    reader = csv.reader(ifile)
    rownum = 0
    for row in reader:
        if rownum == 0:
             rownum += 1
        else:
            x_coordinates.append(float(row[0]))
            y_coordinates.append(float(row[1])/(VM_PARAMETER_UNIT))
        rownum += 1
    ifile.close()
    #print(x_coordinates)
    #print(y_coordinates)

    # Regression
    xdata = np.array(x_coordinates)
    ydata = np.array(y_coordinates)

   # popt, pcov = curve_fit(quad,xdata,ydata)
    #print(popt)

    #Plot row data
    plt1.plot(xdata, ydata, '*')

    #plot regression line of data
    #plt1.plot(xdata, quad(xdata,popt[0],popt[1],popt[2], popt[3]), '-')
    #plot EMA
    line2d = plt1.plot(xdata, EMA.ema(ydata,3))

    #Initialize MIN_VMs
    #VM = [23,42] #Todo fill with randoms

    for j in range(0,MIN_VM):
        # id = randint(1,999)
        t = randint(0,59)#take this as arg?
        vm = VM(initTime = -t, endTime= -1)
        listVM.append(vm)

    #Plot number of VMs required
    roundup_required_old = MIN_VM
    for i in drange(0, max(xdata)-SHIFT, 0.1):
        z = getVaue(line2d,i)
        #print(z)
        roundup_required = math.ceil(z/VM_THRESHOLD_PRECENTAGE)
        vm_change = int(math.ceil(roundup_required- roundup_required_old))

        if vm_change>= 0:
            for k in range(0,vm_change):
                startVM(listVM, i)
        else:
            for k in range(0, -vm_change+1):
                listVM = removeVM(listVM, PROVIDER, i)
        if roundup_required < MIN_VM:
            roundup_required = MIN_VM
        roundup_required_old = roundup_required;
        #print(roundup_required)
        digix_cordinates.append(i)
        digiy_cordinates.append(roundup_required)

    digixdata = np.array(digix_cordinates)
    digiydata = np.array(digiy_cordinates)
    plt1.plot(digixdata,digiydata)


    for vm in listVM:
        print("VM_id : %s initTime : %s endTime : %s" % (vm.id, vm.initTime, vm.endTime) )

    costValues = array.array('d')
    for k in drange(0,max(xdata), 10):
        costValues.append(calculateAWSCost(listVM, k,VM_PRICE_PER_HOUR))
    costydata= np.array(costValues)
    costxdata = np.arange(0,max(xdata),10)

    plt2.plot(costxdata,costydata)
    #plt.show()
    return plt
    #for i in range(0, len(x_coordinates)):
    #    print(x_coordinates[i])
run(4,.8,0,2,0,"default",12).show()
run(4,.8,0,2,0,"aws",12).show()

