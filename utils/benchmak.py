import torch
import time
import random
s=time.time()
for i in range(100):
    rnd=random.randint(128//64,1024//64)*64
    torch.Tensor(rnd,rnd)@torch.Tensor(rnd,rnd)
print(time.time()-s)