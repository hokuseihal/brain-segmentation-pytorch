import torch


def cutmix(x, param=None, ratio=0.3):
    B, C, H, W = x.shape
    if param is None:
        px, py = torch.randint(int(H * ratio), int(H * (1 - ratio)), [1]), torch.randint(int(W * ratio),
                                                                                         int(W * (1 - ratio)), [1])
        print('p',px,py)
        mask = torch.zeros(H, W).to(x.device)
        mask[:px, :py] = 1
        if torch.randint(2, [1]) == 0:
            mask = mask.flipud()
        if torch.randint(2, [1]) == 0:
            mask = mask.fliplr()
        randperm = torch.randperm(B)
        # randperm=torch.tensor([1,0])
    else:
        mask, randperm = param
    # x[..., mask] = x[randperm][..., mask]
    x=x*mask+x[randperm]*(1-mask)
    return x, (mask, randperm)


if __name__ == '__main__':
    x = torch.ones(2, 1, 8, 8)
    x[1]=2
    y ,param= cutmix(x)
    print(y)
    z,_=cutmix(x,param)
    print(z)
