import pickle
import numpy as np
import sys

with open(f'data/{sys.argv[1]}/data.pkl','rb') as f:
    data=pickle.load(f)
print(np.mean(data['acc:miou'][-20:]))