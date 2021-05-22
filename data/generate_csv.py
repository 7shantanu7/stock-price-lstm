'''Script to generate NSE symbol list and output in csv format'''

import csv
import pandas as pd
from nsetools import Nse

nse = Nse()
list = []

all_stock_codes = nse.get_stock_codes()
with open('outputf.csv', 'w', newline='') as output:
    writer = csv.writer(output)
    for key, value in all_stock_codes.items():
        writer.writerow([key, value])

#for nse symbols
list = []
with open('outputf.csv', newline='') as csvfile:
    data = csv.DictReader(csvfile)
    for row in data:
        list.append(row['SYMBOL'])

list2=[]                
for i in list:
    i+=".NSE"
    list2.append(i)

print(list2)
