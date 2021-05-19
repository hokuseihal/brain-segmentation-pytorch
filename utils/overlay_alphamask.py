from PIL import Image
import os
import numpy as np
def overlay(img_p,mask_p,alpha=128):
    img=Image.open(img_p)
    img.putalpha(255)
    mask=Image.open(mask_p)
    mask.putalpha(alpha)
    mask=np.array(mask)
    mask[(mask[...,:3]==0).sum(axis=-1)==3,-1]=0
    mask=Image.fromarray(mask)
    ret=Image.alpha_composite(img,mask)
    ret.show()

if __name__=='__main__':
    overlay(f'{os.environ["HOME"]}/src/data/owncrack/scene/image/IMG_20200517_182522.jpg',f'{os.environ["HOME"]}/src/data/owncrack/scene/mask/IMG_20200517_182522.jpg')