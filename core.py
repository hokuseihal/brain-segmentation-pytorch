import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle
import torch

def addvalue(dict,key,value,epoch):
    if not key in dict.keys():
        dict[key]=[value]
    else:
        if epoch > len(dict[key])-1:
            dict[key].append(value)
        else:
            dict[key][epoch]=value
def savedic(dict,fol):
    n=1
    numgraph=len(set([i.split(':')[0] for i in dict]))
    axdic={}
    fig=plt.figure()
    for key in dict:
        graph,label=key.split(':')
        if graph in axdic:
            axdic[graph].plot(dict[key],label=f'{graph}:{label}')
        else:
            axdic[graph]=fig.add_subplot(numgraph,1,n)
            n+=1
            axdic[graph].plot(dict[key],label=f'{graph}:{label}')
    for key in axdic:
        axdic[key].legend()
    #fig.legend()
    fig.savefig(f'{fol}/graphs.png')
    with open(f'{fol}/data.pkl','wb') as f:
        pickle.dump(dict,f)

def save(e,model,fol,dic=None):
    savedmodelpath=f'{fol}/model.pth'
    if dic:
        try:
            savedic(dic,'/'.join(savedmodelpath.split('/')[:-1]))
        except:
            print('cannnot save fig.')
    torch.save(model.state_dict(), savedmodelpath)
    with open(f'{fol}/.epoch','w') as f:
        f.write(f'{e}')
