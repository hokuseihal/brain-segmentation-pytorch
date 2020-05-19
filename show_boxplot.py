import pickle
import matplotlib.pyplot as plt
import sys
with open(f'{sys.argv[1]}/data.pkl','rb') as f:
    data=pickle.load(f)
plt.boxplot(data['acc:miou'])
plt.savefig('boxplot.png')