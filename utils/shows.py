from PIL import Image
import glob
import random
import time
from core import load
saved = load('data/crack_report/normal_x')
writer, preepoch, modelpath, worter = saved['writer'], saved['epoch'], saved['modelpath'], saved['worter']
trainmask, validmask = worter['trainmask'], worter['validmask']
import os
outfolder='data/out/cat'
os.makedirs(outfolder,exist_ok=True)
for i in range(len(validmask)):
    mask1 = f'/home/hokusei/src/crack-segmentation/data/out/normal_x/{i}_out1.png'
    mask2 = f'/home/hokusei/src/crack-segmentation/data/out/focal2_x/{i}_out1.png'
    mask3 = f'/home/hokusei/src/crack-segmentation/data/out/split2_x/{i}_out.png'
    mask4 = f'/home/hokusei/src/crack-segmentation/data/out/split2focal_x/{i}_out.png'
    dst=Image.new('RGB',(256*6+5,256),(256,256,256))
    dst.paste(Image.open(validmask[i].replace('mask','image')).resize((256,256)))
    dst.paste(Image.open(validmask[i]).resize((256,256)),(256+1,0))
    dst.paste(Image.open(mask1).resize((256, 256)),(256*2+2,0))
    dst.paste(Image.open(mask2).resize((256, 256)),(256*3+3,0))
    dst.paste(Image.open(mask3).resize((256, 256)),(256*4+4,0))
    dst.paste(Image.open(mask4).resize((256, 256)),(256*5+5,0))
    dst.save(f'{outfolder}/{i}.png')
    print(i)
