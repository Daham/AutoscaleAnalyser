__author__ = 'bhash90'

import numpy as np


def violation_precentage(machine_count,machine_unit_power, predicted_arr ):
    tot_power = machine_count * machine_unit_power
    count = 0.0
    for x in predicted_arr :
        if(x > tot_power):
            count = count +1
    violation = (float)(count) / (float)(len(predicted_arr))

    return violation*100


def minToMaxIteration(minVM, maxVM, machine_unit_power,machine_unit_price, predicted_arr, time_gap ):
    tot_cost = 999999
    time = time_gap * (float)(len(predicted_arr))/60.0
   # print(time)
    index =   minVM
    violation_selected = 0
    for i in range(minVM, maxVM+1):
        precentage = violation_precentage(i,machine_unit_power,predicted_arr)
        cost =  i*machine_unit_price*time + i*machine_unit_price*time*SLA_func(precentage)
        #print("VM: %d  Cost: %d, Violation : %d"  %(i, cost, precentage))
        if cost < tot_cost :
            tot_cost = cost
            index = i
            violation_selected = precentage
        if precentage == 0:
            break
    return index, precentage

def SLA_func(precenage):
    violation_cost = precenage/30.0
    #print(violation_cost)
    return  violation_cost

arr_2d = np.genfromtxt("predict2.csv", delimiter= ",")
prev = -1 #time of last dataset
prev_index = 0
cost = 0;
changed = False
for arr in arr_2d :
    #print(arr[np.logical_not(np.isnan(arr))]) # remove nan
    current = arr[0]
    time_gap = 0

    if prev != -1:
        time_gap = current -prev

    if changed and time_gap < 2:
        continue

    MIN_VM = 2
    MAX_VM = 15
    MACHINE_UNIT_POWER = 12
    MACHINE_UNIT_PRICE = 120
    TIME_GAP = 0.2
    index, violation = minToMaxIteration(MIN_VM,MAX_VM, MACHINE_UNIT_POWER, MACHINE_UNIT_PRICE, arr[1:len(arr)], TIME_GAP)
    if index != prev_index:
        changed = True
    cost  += index * time_gap/60.0 + index * time_gap/60.0* SLA_func(violation)
    prev = current;
    prev_index = index;
    print("Index: %d, Cost : %.3f" %(prev,cost))