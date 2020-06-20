from core import load
def loader(folder):
    saved=load(folder)
    validmask=saved['worter']['validmask']
    return validmask

if __name__=="__main__":
    mask1=loader('data/normal1')
    mask2=loader('data/focal2_1')
    print(sorted(mask1)[:10])
    print(sorted(mask2)[:10])
    print(set(mask1)==set(mask2))
