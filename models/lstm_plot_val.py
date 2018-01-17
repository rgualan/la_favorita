import numpy as np
import pickle
import matplotlib.pyplot as plt


yv = pickle.load(open('../output/lstm_yv.pickle', 'rb'))
yv_pred = pickle.load(open('../output/lstm_val_pred.pickle', 'rb'))

plt.figure(figsize=(15, 6))
plt.plot(np.mean(yv, axis=0), label='Val_observations')
plt.plot(np.mean(yv_pred, axis=0), label='Val_predictions')
plt.savefig('../output/lstm_val.png')
plt.show()