__author__ = 'bhash90'

import numpy as np
import array

def violation_precentage(machine_count,machine_unit_power, predicted_arr ):
    tot_power = machine_count * machine_unit_power
    count = 0.0
    for x in predicted_arr :
        if(x > tot_power):
            count = count +1
    violation = (float)(count) / (float)(len(predicted_arr))
    return violation*100


def minToMaxIteration(minVM, maxVM, machine_unit_power,machine_unit_price, predicted_arr, time_gap ):
    predicted_arr2 = predicted_arr[np.logical_not(np.isnan(predicted_arr))]
    tot_cost = 999999
    time = time_gap * len(predicted_arr2)/60.0

    # print(time)
    index =   minVM
    violation_selected = 0
    for i in range(minVM, maxVM+1):
        precentage = violation_precentage(i,machine_unit_power,predicted_arr2)
        cost =  i*machine_unit_price*time + i*machine_unit_price*time*SLA_func(precentage)
        print("VM: %d  Cost: %d, Violation : %d"  %(i, cost, precentage))
        if cost < tot_cost :
            tot_cost = cost
            index = i
            violation_selected = precentage
        if precentage == 0:
            break
    return index, violation_selected

def SLA_func(precenage):
    violation_cost = precenage/5.0
    #print(violation_cost)
    return  violation_cost

def run(filename):

    arr_2d = np.genfromtxt(filename, delimiter= ",")

    digix_cordinates = []
    digiy_cordinates = []
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
            digix_cordinates.append(current)
            digiy_cordinates.append(prev_index)
            continue

        MIN_VM = 2
        MAX_VM = 100
        MACHINE_UNIT_POWER = 4
        MACHINE_UNIT_PRICE = 6
        TIME_GAP_IN_PREDICTION = 1
        index, violation = minToMaxIteration(MIN_VM,MAX_VM, MACHINE_UNIT_POWER, MACHINE_UNIT_PRICE, arr[1:len(arr)], TIME_GAP_IN_PREDICTION)
        if index != prev_index:
            changed = True
        newcost  = index * time_gap/60.0 + index * (time_gap/60.0)* SLA_func(violation)
        cost += newcost
        prev = current;
        prev_index = index;
        digix_cordinates.append(current)
        digiy_cordinates.append(index)
        print("VM: %d Violation : %.3f Time: %d, newCost: %.3f Cost : %.3f" %(prev_index , violation, prev, newcost,cost))
    return  digix_cordinates, digiy_cordinates

#RUN
#run("../simulation/data/predicted.csv")
