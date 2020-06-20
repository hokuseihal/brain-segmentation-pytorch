import torch
import time
import random
s=time.time()
device='cuda'
for i in range(10):
    rnd=random.randint(4096//64,8192//64)*64
    torch.Tensor(rnd,rnd).to(device)@torch.Tensor(rnd,rnd).to(device)
print(time.time()-s)