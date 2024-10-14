import numpy as np
from matplotlib import pyplot as plt

pred = np.load('pred_nilm.npy')
true = np.load('true_nilm.npy')
for i in range(0,50,2):
    plt.plot(pred[i][:,4])
    plt.plot(true[i][:,4])
    plt.show()