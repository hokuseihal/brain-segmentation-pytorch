from core import load
savefolder='data/crack_report/normal_x'
saved = load(savefolder)
writer, preepoch, modelpath, worter = saved['writer'], saved['epoch'], saved['modelpath'], saved['worter']
trainmask, validmask = worter['trainmask'], worter['validmask']

savefolder='data/crack_report/split2focal_x'
saved = load(savefolder)
writer1, preepoch1, modelpath1, worter1 = saved['writer'], saved['epoch'], saved['modelpath'], saved['worter']
trainmask1, validmask1 = worter1['trainmask'], worter['validmask']
print(validmask1==validmask)
