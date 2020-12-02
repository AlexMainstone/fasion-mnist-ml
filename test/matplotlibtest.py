from matplotlib import pyplot as plt
import random
import time
import numpy as np

first = [0]
second = [0]
third = [0]

firstval = 0
secondval = 0
thirdval = 0

step = 0
maxstep = 5

plt.ion()
while True:
    step+=1
    if random.randint(0, 5) < 2: firstval += 1
    if random.randint(0, 5) < 2: secondval += 1
    if random.randint(0, 5) < 2: thirdval += 1
    
    if step >= maxstep:
        time.sleep(0.1)
        first.append(first[-1]+firstval)
        second.append(second[-1]+secondval)
        third.append(third[-1]+thirdval)
        
        print("yo")
        plt.clf()
        plt.plot(first)
        plt.draw()
        plt.show()
        