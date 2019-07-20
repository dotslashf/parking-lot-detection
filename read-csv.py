import pandas as pd
import csv
import time

filename = "car.csv"

with open(filename, "r") as f:
    reader = csv.reader(f,delimiter = ",")
    data = list(reader)
    row_count = len(data)
    header = data[0]

#data.list_datasets()

while True:
    for data in pd.read_csv(filename, skiprows = row_count - 10, names=["Date", "NO ID", "X Y", "XEND YEND"]):
        data.list_datasets()
    time.sleep(1)