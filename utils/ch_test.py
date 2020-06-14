import glob
import cv2
images='/home/hokusei/src/data/owncrack/scene/mask/*.jpg'
for imgp in glob.glob(images):
    im=cv2.imread(imgp)
    print(im.shape)
    assert im.shape[-1]==3,f'{imgp}'