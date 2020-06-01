from PIL import Image
import glob
folder='../data/owncrack/scene/image'
savefolder='../data/owncrack/scene/image/patch'
sW=15
sH=10
for imgp in glob.glob(f'{folder}/*.jpg'):
    print(imgp)
    im=Image.open(imgp)
    w=im.size[0]//sW
    h=im.size[1]//sH
    for x_idx,x0 in enumerate(range(0, im.size[0], w)):
        for y_idx,y0 in enumerate(range(0, im.size[1], h)):
            im.crop((x0, y0, x0 + w, y0 + h)).save(imgp.replace(folder,savefolder).replace('.jpg',f'{x_idx}_{y_idx}.jpg'))