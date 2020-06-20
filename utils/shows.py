from PIL import Image
import glob
import random
import time
masks = glob.glob(f'/home/hokusei/src/data/owncrack/scene/mask/*.jpg')
random.seed(0)
k_shot = int(len(masks) * 0.8)
trainmask = random.sample(masks, k=k_shot)


validmask = sorted(list(set(masks) - set(trainmask)))
mask1 = sorted(glob.glob('/home/hokusei/src/crack-segmentation/data/out/normal1/*_out1.png'))
mask2 = sorted(glob.glob('/home/hokusei/src/crack-segmentation/data/out/focal2_1/*_out1.png'))
mask3 = sorted(glob.glob('/home/hokusei/src/crack-segmentation/data/out/split2_1/*_out.png'))
mask4 = sorted(glob.glob('/home/hokusei/src/crack-segmentation/data/out/split2focal_1/*_out.png'))

for i in range(len(validmask)):
    dst=Image.new('RGB',(256*5,256))
    # dst.paste(Image.open(validmask[i]).resize((256,256)))
    dst.paste(Image.open(mask1[i]).resize((256, 256)),(256,0))
    dst.paste(Image.open(mask2[i]).resize((256, 256)),(256*2,0))
    dst.paste(Image.open(mask3[i]).resize((256, 256)),(256*3,0))
    dst.paste(Image.open(mask4[i]).resize((256, 256)),(256*4,0))
    print(i)
    dst.show()
    time.sleep(3)
