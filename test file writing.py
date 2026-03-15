import os
import csv
a = "hello"
b = "world"
c = "!"
filename = "blinks.csv"

if not os.path.exists(filename):
    with open(filename, mode = 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['first', 'second', 'third'])



with open(filename, mode = 'a', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow([a,b,c])

print("Print Success")