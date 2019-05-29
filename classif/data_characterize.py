

import numpy as np
import scipy as sp
import scipy.stats
import pickle
## Import the packages
# from scipy import stats
import matplotlib.pyplot as plt

path = 'logs/pics/model_feat_DA_others/model_feat_DA_others_event_durations.pickle'
var_lens = pickle.load(open(path, 'rb'))


n, bins, patches = plt.hist(var_lens['overall'], 62, facecolor='green', alpha=0.5)
plt.xlabel('duration [s]')
plt.ylabel('number of events')
# plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.title('overall')
plt.axis([0, 3.5, 0, 1500])
plt.grid(True)
plt.show()


n, bins, patches = plt.hist(var_lens['overall'], 62, facecolor='blue', alpha=0.5, cumulative=True, density=True)
plt.xlabel('duration [s]')
plt.ylabel('percentage of events')
# plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.title('overall - cumulative')
plt.axis([0, 3.5, 0, 1.1])
plt.grid(True)
plt.show()


a=9
