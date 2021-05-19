import glob
import pickle as pkl

import matpltolib.pyplot as plt
import numpy as np

root = 'saved'
root = 'saved_thresh'
max_f1 = 0
mat = np.zeros(10 * 10)
for idx, p in enumerate(glob.glob(f'{root}/*.pkl')):
    pred_id = int(p.split('/')[-1].split('.')[0])
    if pred_id % 2 == 0:
        with open(p, 'rb') as f:
            data = pkl.load(f)
            f1 = data[0][2]
            print(f1)
            mat[pred_id // 2] = f1
            if max_f1 < f1:
                max_data = data
                max_f1 = f1
print(max_f1)
print(max_data)

plt.imshow(mat.reshape(10, 10))
plt.show()
