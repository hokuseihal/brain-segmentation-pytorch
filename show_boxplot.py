import pickle
import matplotlib.pyplot as plt
import sys
folder=sys.argv[1]
if folder[-1]!='/':
    folder=folder+'/'
with open(f'{sys.argv[1]}/data.pkl','rb') as f:
    data=pickle.load(f)
plt.boxplot(data['acc:miou'])
plt.savefig(f'{sys.argv[1]}/boxplot.png')