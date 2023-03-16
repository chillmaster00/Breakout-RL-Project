import matplotlib.pyplot as plt
import numpy as np

x = np.array(['A', 'B', 'C', 'D', 'E'])
y = np.array([0.96, 0.97, 0.98, 0.99, 1.0])

plt.bar(x, y)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('My Bar Chart')
plt.tight_layout()
plt.show()
        # epsilonMin = 0.01
        # epsilonMax = 1
        # epsilonDecay = 1000000
        
            # (i + epOffset)
            # eps = max(epsilonMin, epsilonMax - (epsilonMax - epsilonMin) * (i)/ epsilonDecay)