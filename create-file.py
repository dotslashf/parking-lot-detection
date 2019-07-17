from datetime import datetime
import time

f = open("test.txt", "w+")

for i in range(5):
    f.write(str(i))
    f.write("\n")
    time.sleep(1)

f.close()