__author__ = 'bhash90'
import numpy as np;
from matplotlib.lines import Line2D

def avg(values, i):
    return (values[i]+ values[i-1]+ values[i-2])/3

def Stratos(xvalues):
    tempx = xvalues
    u = []
    a = []
    predicted = []
    for i in drange(2, len(xvalues), 1):
        t = 1
        u = (tempx[i] - tempx[i-2])/2.0
        #print("U: %s" %u)
        tempu_1 = (tempx[i] - tempx[i-1])
        #print("U1: %s" %tempu_1)
        tempu_2 = (tempx[i-1] - tempx[i-2])
        #print("U2: %s" %tempu_2)
        a = (tempu_1 - tempu_2)
        #print("a: %s" %a)
        s = u*t + 0.5*a*t*t
        #print("s: %s" %s)
        #print(avg(xvalues,i))
        prediction = s + xvalues[i]
        predicted.append(prediction)
    return predicted

def drange(x, y, jump):
    while x < y:
        yield x
        x += jump

arr = [66.12,70.49,66.98,68.15,71.98,73.52,74.43,66.14,66.92,69.36,72.00,96.29,99.55,78.10,73.98,73.20,74.38,73.89,
73.72,74.42,71.83,70.59,66.52,71.52,65.71,65.38,96.11]
#predict = Stratos(arr)
#print(predict)
