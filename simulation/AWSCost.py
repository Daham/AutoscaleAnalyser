from simulation import AWS

__author__ = 'bhash90'
import csv
import array


def read():
    #type  = array.array("str")
    memory = array.array("d")
    vCpu = array.array("d")
    ECU = array.array("d")
    unix_pricing = array.array("d")
    ifile = open('aws_pricing.csv', "rb")
    reader = csv.reader(ifile, delimiter = "\t")
    print(reader)
    for row in reader:
            print( row)
            #type.append(row[0])
            vCpu.append(float(row[1]))
            ECU.append(float(row[2]))
            memory.append(float(row[3]))
            unix_pricing.append(float(row[4]))
    ifile.close()
    return [memory,vCpu,ECU,unix_pricing]

valueList = read()
for i in  range(0,len(valueList[0])):
    plt,e = AWS.run(valueList[1][i], 0.8, 5, 2, 0, "aws", valueList[3][i],[20,40])
    plt.show()

