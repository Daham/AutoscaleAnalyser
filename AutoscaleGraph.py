__author__ = 'bhash90'
#Todo Take parameters in VM_PARAMETER_UNIT,VM_THRESHOLD_PRECENTAGE,PROVIDER etc as user args
#Todo  implement smart killing under removeVM
#Todo  implement cost for each provider by iterate through listVM
import math
import csv
import array
import matplotlib.pyplot as plt
import numpy as np;
from scipy.optimize import curve_fit
from random import randint

def quad(x, a, b, c):
     return a * x*x+ b*x + c

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

listVM = list()

def startVM(i):
    vm = VM(initTime = i, endTime= -1)
    listVM.append(vm)

#random remove implemented. Check he provider and  implement as a switch
def removeVM(provider, j):
    #Todo  implement smart killing under removeVM
    if(provider == "default"):
        for  vm in listVM:
            if(vm.endTime == -1):
                vm.endTime = j
                print("End VM %s at %s" % (vm.id, j))
                break

VM_PARAMETER_UNIT = 4 #size ofthe parameter in a single node
VM_THRESHOLD_PRECENTAGE = .8 #threshold level defined by user
VM_UPTIME = 0 #how long does it take to spin up an instance
MIN_VM = 2
SHIFT = 0
PROVIDER = "default" #read as user arg

x_coordinates = array.array('d')
y_coordinates =array.array('d')
digix_cordinates = array.array('d')
digiy_cordinates = array.array('d')

#Read vlaues while converting to VM_UNITS
ifile  = open('data.csv', "rb")
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
popt, pcov = curve_fit(quad,xdata,ydata)
print(popt)

#Plot row data
plt.plot(xdata, ydata, '*')

#plot regression line of data
plt.plot(xdata, quad(xdata,popt[0],popt[1],popt[2]), '-')

#Initialize MIN_VMs
#VM = [23,42] #Todo fill with randoms

for j in range(0,MIN_VM):
    # id = randint(1,999)
    t = randint(0,59)#take this as arg?
    vm = VM(initTime = -t, endTime= -1)
    listVM.append(vm)

#Plot number of VMs required
roundup_required_old = MIN_VM
for i in drange(min(xdata), max(xdata)-SHIFT, 0.1):
    z = quad(i+SHIFT,popt[0],popt[1],popt[2])
    print(z)
    roundup_required = math.ceil(z/VM_THRESHOLD_PRECENTAGE)
    vm_change = int(math.ceil(roundup_required- roundup_required_old))

    if vm_change>= 0:
        for k in range(0,vm_change):
            startVM(i)
    else:
        for k in range(vm_change,-1):
         removeVM(PROVIDER, i)
    if roundup_required < MIN_VM:
        roundup_required = MIN_VM
    roundup_required_old = roundup_required;
    print(roundup_required)
    digix_cordinates.append(i)
    digiy_cordinates.append(roundup_required)

digixdata = np.array(digix_cordinates)
digiydata = np.array(digiy_cordinates)
plt.plot(digixdata,digiydata)


for vm in listVM:
    print("VM_id : %s initTime : %s endTime : %s" % (vm.id, vm.initTime, vm.endTime) )

plt.show()
#for i in range(0, len(x_coordinates)):
#    print(x_coordinates[i])

