from distutils.command.upload import upload

__author__ = 'bhash90'
#Todo Graphs should be in same figure where necessary
#Todo Name Axes and lines
import math
import csv
import array
import matplotlib.pyplot as plt
import numpy as np
import EMA
from cost_model import CostModel
from scipy.integrate import simps
from matplotlib.lines import Line2D
import Stratos

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
               # print("Blind Kill from Provider................................................................ : %s" %provider)
                candidates.append((vm, 0))	#dummy value for hourFraction
                if len(candidates) == count:
                    break
            elif provider == "aws":		#first pick all candidates
               # print("Smart Kill from Provider : %s" %provider)
                hourFraction = (now - vm.initTime)%60
                if AWS_MIN_MINUTES_BEFORE_KILL < hourFraction < AWS_MAX_MINUTES_BEFORE_KILL:
                    candidates.append((vm, hourFraction))
            #else:
                # print(".................................................................")

    if provider == "aws":		#sort descending w.r.t. hourFraction
        candidates.sort(lambda x, y: (int)(y[1] - x[1]))

    #kill as many as possible/required
    for (vm, hourFraction) in candidates:
        vm.endTime = now
        print("End VM %s at %s" % (vm.id, now))
        killed = killed + 1
        if killed == count:
            break
    #print("%s Killed %d" %(provider, killed))
    return killed

def calculateAWSCost(listVM, givenTime, price_per_hour):
    cost = 0
    for vm in listVM:
        if(vm.initTime < givenTime):
            if(vm.endTime == -1 or (vm.endTime > givenTime)):
                billing_hours = int(math.ceil((givenTime - vm.initTime)/60.0))
                cost += billing_hours * price_per_hour
                ##print("cost: %s at %s and %s init %s" % (cost,givenTime,vm.endTime, vm.initTime))
            else:
                billing_cycle_minutes = (vm.endTime - vm.initTime)% 60
                if billing_cycle_minutes< 57.0:
                    billing_hours = int(math.ceil((vm.endTime - vm.initTime)/ 60))
                    cost += billing_hours * price_per_hour
                else: # blind killing after 57 minutes will be charged for next hour
                    billing_hours = int(math.ceil((vm.endTime - vm.initTime) /60 ))
                    cost += (billing_hours +1) * price_per_hour
                #print("cost: %s at %s" % (cost,givenTime))
    return cost

def getValue(line2d, x):
    xvalues = line2d.get_xdata()
    yvalues = line2d.get_ydata()
    idx = np.where(xvalues == xvalues[x])
    return yvalues[idx[0]]

def mse(predictions, targets):
    e = np.sqrt(np.mean((predictions - targets)**2))
    return e

def run(VM_parameter_unit, threshold_percentage, uptime, min_VM, shift, provider, prediction_type,
	vm_price_per_hour, vm_init_data, actualFileName, scaleFileName, costFileName):
    listVM = list()

    open(scaleFileName, 'w').close()
    open(costFileName, 'w').close()
    scaleCSV = open(scaleFileName,"rw+")
    costCSV = open(costFileName,"rw+")

    scaleCSV.write("Time,VM Count\n" )
    costCSV.write( "Time,Total Cost\n" )

    #print(provider)

    digix_cordinates = array.array('d')
    digiy_cordinates = array.array('d')
    x_coordinates = array.array('d')
    y_coordinates = array.array('d')
    digiy_coord_actual = array.array('d')
    modely_coord_actual = array.array('d')

    #Read vlaues while converting to VM_UNITS
    ifile = open(actualFileName, "rb")
    reader = csv.reader(ifile)
    rownum = 0
    for row in reader:
      x_coordinates.append(float(row[0]))
      y_coordinates.append(float(row[1])/(VM_parameter_unit))
      rownum += 1
    ifile.close()
    ##print(x_coordinates)
    ##print(y_coordinates)

    # Regression
    xdata = np.array(x_coordinates)
    ydata = np.array(y_coordinates)

    #Plot row data
    rowdata = Line2D(xdata, ydata)

    #plot regression line of data
    #plt1.plot(xdata, quad(xdata,popt[0],popt[1],popt[2], popt[3]), '-')
    #plot EMA
    line2d = Line2D(xdata, EMA.ema(ydata ,3))

    #plot stratos
    line2d_stratos = Line2D(xdata[2:len(xdata)]+1,Stratos.Stratos(ydata))

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
    if(provider == "default" and prediction_type == "reactive"):
        for i in drange(uptime, max(xdata) - shift + uptime- 1, 0.1):
            #print(i-uptime)
            z = getValue(line2d, i - uptime)
            yvalueset.append(z)
            ##print(z)
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
            scaleCSV.seek(0, 2)
            scaleCSV.write("%.3f,%.3f\n" %(i, vm_count*VM_parameter_unit))
        digixdata = np.array(digix_cordinates)
        digiydata = np.array(digiy_cordinates)
        actualydata = np.array(digiy_coord_actual)
        lineAllocate = Line2D(digixdata, digiydata) #requirement
        digi_line  = Line2D(digixdata, actualydata) #actual

    elif(provider == "default" and prediction_type == "stratos"):
        request_count = 0
        sampling_distance = 0.1
        for i in drange(uptime, max(xdata) - shift + uptime -4, sampling_distance):
            print(i)
            line2d = line2d_stratos
            z = getValue(line2d_stratos, i - uptime)
            yvalueset.append(z)
            ##print(z)
            new_vm_count = math.ceil(z/threshold_percentage)
            if new_vm_count < min_VM:
                new_vm_count = min_VM

            vm_change = int(math.ceil(new_vm_count - vm_count))
            if vm_change > 0:
                vm_count += startVMs(listVM, vm_change, i)
                request_count = 0
            elif vm_change < 0:
                request_count += 1
                if request_count > 2.0/sampling_distance :
                    vm_count -= removeVMs(listVM, -vm_change, provider, i)
                    request_count = 0

            digix_cordinates.append(i)
            digiy_cordinates.append(new_vm_count)
            digiy_coord_actual.append(vm_count)
            scaleCSV.seek(0, 2)
            scaleCSV.write("%.3f,%.3f\n" %(i, vm_count*VM_parameter_unit))
        digixdata = np.array(digix_cordinates)
        digiydata = np.array(digiy_cordinates)
        actualydata = np.array(digiy_coord_actual)
        lineAllocate = Line2D(digixdata, digiydata) #requirement
        digi_line  = Line2D(digixdata+4, actualydata) #actual

    elif(provider == "aws" and prediction_type == "stratos"):
        request_count = 0
        sampling_distance = 0.1
        for i in drange(uptime, max(xdata) - shift + uptime -4, 0.1):
            print(i)
            line2d = line2d_stratos
            z = getValue(line2d, i - uptime)
            yvalueset.append(z)
            ##print(z)
            new_vm_count = math.ceil(z/threshold_percentage)
            if new_vm_count < min_VM:
                new_vm_count = min_VM

            vm_change = int(math.ceil(new_vm_count - vm_count))
            if vm_change > 0:
                vm_count += startVMs(listVM, vm_change, i)
                request_count = 0
            elif vm_change < 0:
                request_count += 1
                if request_count > 2.0/sampling_distance:
                    vm_count -= removeVMs(listVM, -vm_change, provider, i)

            digix_cordinates.append(i)
            digiy_cordinates.append(new_vm_count)
            digiy_coord_actual.append(vm_count)
            scaleCSV.seek(0, 2)
            scaleCSV.write("%.3f,%.3f\n" %(i, vm_count*VM_parameter_unit))
        digixdata = np.array(digix_cordinates)
        digiydata = np.array(digiy_cordinates)
        actualydata = np.array(digiy_coord_actual)
        lineAllocate = Line2D(digixdata, digiydata) #requirement
        digi_line  = Line2D(digixdata+4, actualydata) #actual

    elif(provider == "aws" and prediction_type == "reactive"):
            for i in drange(uptime, max(xdata) - shift + uptime- 1, 0.1):
                #print(i-uptime)
                z = getValue(line2d, i - uptime)
                yvalueset.append(z)
                ##print(z)
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
                scaleCSV.seek(0, 2)
                scaleCSV.write("%.3f,%.3f\n" %(i, vm_count*VM_parameter_unit))
            digixdata = np.array(digix_cordinates)
            digiydata = np.array(digiy_cordinates)
            actualydata = np.array(digiy_coord_actual)
            lineAllocate = Line2D(digixdata, digiydata) #requirement
            digi_line  = Line2D(digixdata, actualydata) #actual

    elif(provider == "default" and prediction_type == "proactive"):
        digix, digiy = CostModel.run(actualFileName)
        for i in range(0, len(digix)):
            new_vm_count = digiy[i]
            if i != 0:
                vm_change = int(math.ceil(digiy[i] - len([i for x in listVM if x.endTime == -1])))
            else :
                vm_change = int(math.ceil(digiy[0] - len([i for x in listVM if x.endTime == -1])))

            if vm_change > 0:
                vm_count += startVMs(listVM, vm_change, digix[i])
            elif vm_change < 0:
                vm_count -= removeVMs(listVM, -vm_change, provider, digix[i])
            modely_coord_actual.append(vm_count)
            scaleCSV.seek(0, 2)
            scaleCSV.write("%.3f,%.3f\n" %(i, vm_count*VM_parameter_unit))

        digixdata = np.array(digix)
        digiydata = np.array(digiy)
        lineAllocate = Line2D(digixdata, digiydata) #requirement
        digi_line  = Line2D(digixdata, modely_coord_actual) #actual

    elif(provider == "aws" and prediction_type == "proactive"):
        digix, digiy = CostModel.run(actualFileName)
        #print(digix)
        #print(digiy)
        for i in range(0, len(digix)):
            new_vm_count = digiy[i]
            if i != 0:
                vm_change = int(math.ceil(digiy[i] - len([i for x in listVM if x.endTime == -1])))
            else :
                vm_change = int(math.ceil(digiy[0] - len([i for x in listVM if x.endTime == -1])))

            if vm_change > 0:
                vm_count += startVMs(listVM, vm_change, digix[i])
            elif vm_change < 0:
                vm_count -= removeVMs(listVM, -vm_change, provider, digix[i])
            modely_coord_actual.append(vm_count)
            scaleCSV.seek(0, 2)
            scaleCSV.write("%.3f,%.3f\n" %(i, vm_count*VM_parameter_unit))

        digixdata = np.array(digix)
        digiydata = np.array(digiy)
        lineAllocate = Line2D(digixdata, digiydata) #requirement
        digi_line  = Line2D(digixdata, modely_coord_actual) #actual

    #for vm in listVM:
        #print("VM_id %3s: %5.1f - %5.1f = %5.1f%s" % (vm.id, vm.initTime, vm.endTime,(vm.endTime if vm.endTime > 0 else i) - vm.initTime, ("" if vm.endTime > 0 else " up")))

    costValues = array.array('d')
    for k in drange(0, max(xdata), 1):
        cost = calculateAWSCost(listVM, k, vm_price_per_hour)
        costValues.append(cost)
        costCSV.seek(0, 2)
        costCSV.write("%.3f,%.3f\n" %(k,cost))

    costydata = np.array(costValues)
    costxdata = np.arange(0, max(xdata), 1)

    cost_line = Line2D(costxdata,costydata)

    yvalues = np.array(yvalueset)
    #e = mse(digiydata, yvalues)

    start = max(min(line2d.get_xdata()), min(lineAllocate.get_xdata()))
    end   = min(max(line2d.get_xdata()), max(lineAllocate.get_xdata()));
    #calculateViolation(predictLine=line2d, allocateline=lineAllocate, startTime= start , endTime= end)
    return rowdata, line2d, digi_line,cost_line

def calculateViolation(predictLine, allocateline, startTime ,endTime):
    stepSize = 1
    violateArea = 0
    violateTime = 0
    area_violations = []
    time_violations  = []
    for i in  drange(startTime + stepSize, endTime, stepSize):
        predicted_i0 = getValue(predictLine,i - stepSize)
        predicted_i1 = getValue(predictLine,i)
        ##print("Predicty=  i0 :%s i1:%s" %(predicted_i0,predicted_i1))
        allocated_i0 = getValue(allocateline, i - stepSize)
        allocated_i1 = getValue(allocateline,i)
        area_under_predicted  = simps(y = [predicted_i0, predicted_i1] , dx = stepSize)
        area_under_allocated  = simps(y = [allocated_i0, allocated_i1] , dx = stepSize)
        ##print("Areas : %s, %s"  %(area_under_predicted,area_under_allocated))
        if area_under_allocated < area_under_predicted:
            violateArea += (area_under_predicted -area_under_allocated)
            violateTime += stepSize
        area_violations.append(violateArea)
        time_violations.append(violateTime)

    area = np.array(area_violations)
    time = np.array(time_violations)
    xvalue =  np.arange(startTime+stepSize, endTime, stepSize)
    f , (plt1, plt2) = plt.subplots(1,2, sharex= True)
    plt1.plot(xvalue,area)
    plt2.plot(xvalue,time)
    #plt.show()
    #print("ViolateArea : %s ViolateTime : %s" %(violateArea, violateTime))
    return  violateArea, violateTime

# virtual machine unit, threshold, uptime, min, shift, provider, per hour cost, initial data[]
#run(4, 100, 0, 0, 0, "aws", 6, [0,0], "data/actual.csv")

M3_MEDIUM_HOURLY_PRICE = 0.067
REACTIVE_THREASHOLD =  0.8
PROACTIVE_THRESHOLD = 1
MIN_VM =2
VM_PARAM_UNIT = 4

#rowdata, predicted, digi_line,cost_line = run(VM_PARAM_UNIT, REATIVE_THREASHOLDE, 0, MIN_VM, 0, "default","reactive", M3_MEDIUM_HOURLY_PRICE, [0,0], "../datasets/predicted_static/predicted.csv", "data/reactive_scale.csv", "data/normal_cost.csv")

f0, (plt1,plt4) = plt.subplots(2,1,sharey=True)
f1, (plt11,plt3 ) = plt.subplots(2,1,sharey=True)
f_stratos, (plt_str_1, plt_str_2 ) = plt.subplots(2,1,sharey=True)
f2, (func_plot) = plt.subplots(1,1)
f3, (plt2) = plt.subplots(1,1)

violation_x = array.array('d')
violation_y = array.array('d')

for n in drange(0, 100, 0.05):
    violation_x.append(n)
    violation_y.append(CostModel.SLA_func(n))

func_plot.plot(violation_x, violation_y)

#filename = "../datasets/predicted_static/predicted.csv"
filename = "data2predicted.csv"
filename2 = "data2actual.csv"
rowdata, predicted, digi_line,cost_line = run(VM_PARAM_UNIT, REACTIVE_THREASHOLD, 0, MIN_VM, 0, "default","reactive", M3_MEDIUM_HOURLY_PRICE, [0,0], filename, "data/reactive_scale.csv", "data/normal_cost.csv")

plt1.plot(rowdata.get_xdata(), rowdata.get_ydata(), "*") #rowdata
plt1.plot(predicted.get_xdata(), predicted.get_ydata())  #EMA predicted
plt1.plot(digi_line.get_xdata(), digi_line.get_ydata())  #Reactive Blind Killing
plt2.plot(cost_line.get_xdata(), cost_line.get_ydata())  #Reactive Blind Killing Cost

rowdata_str_1, predicted_str_1, digi_line_str_1,cost_line_str_1 = run(VM_PARAM_UNIT, REACTIVE_THREASHOLD, 0, MIN_VM, 0, "default","stratos", M3_MEDIUM_HOURLY_PRICE, [0,0], filename2, "data/reactive_scale.csv", "data/normal_cost.csv")

plt_str_1.plot(rowdata_str_1.get_xdata(), rowdata_str_1.get_ydata(), "*") #rowdata
plt_str_1.plot(predicted_str_1.get_xdata(), predicted_str_1.get_ydata())  #Stratos predicted
plt_str_1.plot(digi_line_str_1.get_xdata(), digi_line_str_1.get_ydata())  #Reactive Blind Killing
plt2.plot(cost_line_str_1.get_xdata(), cost_line_str_1.get_ydata())  #Stratos Blind Killing Cost

rowdata_str_2, predicted_str_2, digi_line_str_2,cost_line_str_2 = run(VM_PARAM_UNIT, REACTIVE_THREASHOLD, 0, MIN_VM, 0, "aws","stratos", M3_MEDIUM_HOURLY_PRICE, [0,0], filename2, "data/reactive_scale.csv", "data/normal_cost.csv")

plt_str_2.plot(rowdata_str_2.get_xdata(), rowdata_str_2.get_ydata(), "*") #rowdata
plt_str_2.plot(predicted_str_2.get_xdata(), predicted_str_2.get_ydata())  #Stratos predicted
plt_str_2.plot(digi_line_str_2.get_xdata(), digi_line_str_2.get_ydata())  #Reactive Blind Killing
plt2.plot(cost_line_str_2.get_xdata(), cost_line_str_2.get_ydata())  #Stratos Smart Killing Cost

rowdata11, predicted11, digi_line11,cost_line11 = run(VM_PARAM_UNIT, PROACTIVE_THRESHOLD, 0, MIN_VM, 0, "default","proactive", M3_MEDIUM_HOURLY_PRICE, [0,0], filename, "data/reactive_scale.csv", "data/normal_cost.csv")

plt11.plot(rowdata11.get_xdata(), rowdata11.get_ydata(), "*") #rowdata
#plt11.plot(predicted11.get_xdata(), predicted11.get_ydata())  #EMA predicted
plt11.plot(digi_line11.get_xdata(), digi_line11.get_ydata())  #Proactive Blind Killing
plt2.plot(cost_line11.get_xdata(), cost_line11.get_ydata())  #Proactive Blind Killing Cost

rowdata2, predicted2, digi_line2,cost_line2 = run(VM_PARAM_UNIT, PROACTIVE_THRESHOLD, 0, MIN_VM, 0, "aws", "proactive", M3_MEDIUM_HOURLY_PRICE, [0,0], filename, "data/proactive_scale.csv", "data/optimized_cost.csv")

plt3.plot(rowdata.get_xdata(), rowdata.get_ydata(), "*") #rowdata
plt3.plot(digi_line2.get_xdata(),digi_line2.get_ydata()) #Proactive Smart Killing
plt2.plot(cost_line2.get_xdata(),cost_line2.get_ydata()) #Proactive Smart Killing cost

rowdata3, predicted3, digi_line3,cost_line3 = run(VM_PARAM_UNIT, REACTIVE_THREASHOLD, 0, MIN_VM, 0, "aws", "reactive", M3_MEDIUM_HOURLY_PRICE, [0,0], filename, "data/proactive_scale.csv", "data/normal2_cost.csv")

plt4.plot(rowdata3.get_xdata(), rowdata3.get_ydata(), "*") #rowdata
plt4.plot(predicted3.get_xdata(), predicted3.get_ydata())  #EMA predicted
plt4.plot(digi_line3.get_xdata(),digi_line3.get_ydata()) #Reactive Smart Killing
plt2.plot(cost_line3.get_xdata(),cost_line3.get_ydata()) #Reactive Smart Killing Cost

tot_count = 0.0
vio_count = 0.0
total_costx = array.array('d')
total_costy = array.array('d')
for m in drange(min(rowdata.get_xdata()), max( rowdata.get_xdata())-1, 0.2):

    tot_count += 1
    row_value     = getValue(rowdata, m)
    predict_value = getValue(digi_line2,m)
    revenue_cost  =  getValue(cost_line2, m)
    if(row_value> predict_value):
        vio_count += 1
    precentage = (vio_count/tot_count)*100.0
    print("Vio_Count: %d" %vio_count)
    print("Tot_count : %d" %tot_count)
    print("Precentage: %s" %precentage)
    vio_cost = CostModel.SLA_func(precentage)*M3_MEDIUM_HOURLY_PRICE*(m/60.0)
    tot_cost = revenue_cost + vio_cost
    print("Violation Cost : %.3f .................................." %vio_cost)
    total_costx.append(m)
    total_costy.append(tot_cost)

print("Vio_count: %d" %vio_count)
print("Tot_count : %d" %tot_count)
plt2.plot(total_costx,total_costy)

#rowdata3, predicted3, digi_line3,cost_line3, e = run(4, .98, 5, 2, 0, "default", 6, [20,40])
#plt1.plot(digi_line3.get_xdata(),digi_line3.get_ydata())

plt1.set_xlabel("Time/minutes")
plt1.set_ylabel("VM_Units")
plt1.legend(["Raw Data", "Predicted", "Reactive-Blind Killing" ], loc='upper right')

plt_str_1.set_xlabel("Time/minutes")
plt_str_1.set_ylabel("VM_Units")
plt_str_1.legend(["Raw Data", "Stratos-Predicted", "Stratos-Blind Killing" ], loc='upper right')

plt_str_2.set_xlabel("Time/minutes")
plt_str_2.set_ylabel("VM_Units")
plt_str_2.legend(["Raw Data", "Stratos-Predicted", "Stratos-Smart Killing" ], loc='upper right')

plt11.set_xlabel("Time/minutes")
plt11.set_ylabel("VM_Units")
plt11.legend(["Raw Data", "Proactive-Blind Killing" ], loc='upper right')

plt3.set_xlabel("Time/minutes")
plt3.set_ylabel("VM_Units")
plt3.legend(["Raw Data", "Proactive -Smart Killing"], loc='upper right')

plt4.set_xlabel("Time/minutes")
plt4.set_ylabel("VM_Units")
plt4.legend(["Raw Data", "Predicted" ,"Reactive -Smart Killing"], loc='upper right')

func_plot.set_xlabel("Violated Precentage")
func_plot.set_ylabel("Penalty Factor/ VM_Units")

plt2.set_xlabel("Time/minutes")
plt2.set_ylabel("Cost")
plt2.legend(["Reactive-Blind Killing","Stratos-Blind Killing ","Stratos-Smart Killing","Proactive-Blind Killing","Proactive-Smart Killing", "Reactive-Smart Killing", "Proactive-Smart Killing-Total" ], loc='upper left')
plt.show()
