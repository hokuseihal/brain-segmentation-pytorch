import matplotlib.pyplot as plt
import pickle


datap=['data/crack_report/normal_x/data.pkl','data/crack_report/focal2_x/data.pkl','data/crack_report/split2_x/data.pkl','data/crack_report/split2focal_x/data.pkl']
name=['normal','focal','split2','focalsplit2']
# color=[['#0000cd','#87ceeb'],['#006400','#00ff7f'],['#ff0000','#ff69b4'],['#000000','#c0c0c0']]
color=['#1f77b4','#ff7f0e','#2ca02c','#d62728']
# plt.subplot(2,1,1)
# for n,p,c in zip(name,datap,color):
#     with open(p,'rb') as f:
#         data=pickle.load(f)
#     plt.plot(data['loss:train'],label=f'{n}:train',color=c)
#     plt.plot(data['loss:valid'],label=f'{n}:valid',color=c,linestyle='dotted')
#     plt.xlim(-2,100)
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#
# plt.subplot(2,1,2)
for n,p,c in zip(name,datap,color):
    with open(p,'rb') as f:
        data=pickle.load(f)
    plt.plot(data['mIoU:train'],label=f'{n}:train',color=c)
    plt.plot(data['mIoU:valid'],label=f'{n}:valid',linestyle='dotted',color=c)
    plt.xlim(-2,100)
    plt.ylabel('mIoU')
    plt.xlabel('epoch')
# plt.legend(loc='center left', bbox_to_anchor=(1., .5))
plt.savefig('data/loss_miou.png')
plt.show()
