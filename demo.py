import torch
from unet import UNet
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
def setcolor(idxtendor, colors=torch.tensor([[0, 0, 0], [255, 255, 255], [0, 255, 0]])):
    assert idxtendor.max() + 1 <= len(colors)
    B, H, W = idxtendor.shape
    colimg = torch.zeros(B, 3, H, W).to(idxtendor.device).to(idxtendor.device)
    colors = colors[1:]
    for b in range(B):
        for idx, color in enumerate(colors, 1):
            colimg[b, :, idxtendor[b] == idx] = (color.reshape(3, 1)).to(idxtendor.device).float()
    return colimg

def demo(imgpath='/home/hokusei/src/data/owncrack/img.jpg',modelpath='data/normal_x/model.pth',outpath='out.jpg'):
    img=T.ToTensor()(Image.open(imgpath).resize((256,256))).unsqueeze(0)
    model=UNet(out_channels=3)
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    for p in model.parameters():
        p.requires_grad=False
    out=model(img)
    save_image(setcolor(out.argmax(1)),outpath)

if __name__=='__main__':
    demo()