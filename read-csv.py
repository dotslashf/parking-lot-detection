import time
import csv

with open('parking.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        if row[" no"] == "1":
            print(f'{row["date"]} | Slot Park: {row[" no"]} | Kosong?: {row[" kosong"]} | Jam Isi: {row[" jam isi"]} | Jam Kosong: {row[" jam kosong"]}')
            print("-------------------------------------------------------------------------------------")
        line_count += 1
    print(f'Processed {line_count} lines.')