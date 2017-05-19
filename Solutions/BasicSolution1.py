import csv

path = "C:\\Users\\MILICA\\Desktop\\doctrina-apparatus\\Solutions\\train.csv"

file = open(path, encoding="utf8")

reader = csv.reader(file)

data = [row for row in reader]

print(data)