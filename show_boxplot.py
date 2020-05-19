import pickle
import matplotlib.pyplot as plt

with open('data.pkl','rw') as f:
    data=pickle.load(f)
plt.boxplot(data['acc:miou'])
plt.savefig('boxplot.png')