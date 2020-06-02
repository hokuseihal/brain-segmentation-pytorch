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
def saveworter(worter,key,value):
    worter[key]=value

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
    plt.close()
    with open(f'{fol}/data.pkl','wb') as f:
        pickle.dump(dict,f)

def save(e,model,fol,dic=None,worter=None):
    savedmodelpath=f'{fol}/model.pth'
    if dic:
        savedic(dic,'/'.join(savedmodelpath.split('/')[:-1]))
    torch.save(model.state_dict(), savedmodelpath)
    with open(f'{fol}/.epoch','w') as f:
        f.write(f'{e}')
    if worter:
        with open(f'{fol}/worter.pkl','wb') as worterf:
            pickle.dump(worter,worterf)
import os
def load(folder):
    if os.path.exists(f'{folder}/data.pkl'):
        with open(f'{folder}/data.pkl','rb') as dataf:
           writer= pickle.load(dataf)
    if os.path.exists(f'{folder}/.epoch'):
        with open(f'{folder}/.epoch','r') as epochf:
            epoch=int(epochf.readline())
    if os.path.exists(f'{folder}/worter.pkl'):
        with open(f'{folder}/worter.pkl','rb') as worterf:
            worter=pickle.load(worterf)
    return {'writer':writer,'epoch':epoch,'modelpath':f'{folder}/model.pth','worter':worter}
def load_check(folder):
    if not os.path.exists(f'{folder}/model.pth'):
        print('You want to load previous session, but not saved')
        return False
    else:
        return True
def savefig(pklpath):
    with open(pklpath,'rb') as f:
        writer=pickle.load(f)
        savedic(writer,'/'.join(pklpath.split('/')[:-1]))
