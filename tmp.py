import os
results = os.listdir()
for result in results:
    cmd = "python sample.py --image='png/" + result
    os.system(cmd)
