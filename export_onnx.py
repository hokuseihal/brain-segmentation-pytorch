import torch
import torchvision

from unet import UNet

model=UNet(3,3).cuda()
model.load_state_dict(torch.load('data/data/split2/model.pth'))
# model_int8 = torch.quantization.quantize_dynamic(
#     model,  # the original model
#     {torch.nn.Conv2d,torch.nn.BatchNorm2d},  # a set of layers to dynamically quantize
#     dtype=torch.qint8)
input_names = [ "input1" ]
output_names = [ "output1" ]
dummy_input=torch.randn(1,256,256,3,device='cuda')
torch.onnx.export(model,dummy_input,'data/unet_crack.onnx',verbose=True,input_names=input_names,output_names=output_names)