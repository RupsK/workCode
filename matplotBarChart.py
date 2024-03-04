# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:35:18 2024

@author: h4tec
"""

import matplotlib.pyplot as plt
import numpy as np
import random

plt.style.use('_mpl-gallery')

# make data:
x = 0.5 + np.arange(8)
i = 0
y = []
#y =  [4.8, 5.5, 3.5, 4.6, 6.5, 6.6, 2.6, 3.0]
for i in range(8):
    r= random.randint(1, 8)
    print(r)
    y.append(r)
print(y)   
# plot
fig, ax = plt.subplots()
 
ax.stairs(y,linewidth=2.5)                  # for stpes type graph 

#ax.bar(x, y, width=1, edgecolor="white", linewidth=0.5)           # for bar chart
#ax.stem(x, y)                      # for stem line and point at top end ----.
ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()