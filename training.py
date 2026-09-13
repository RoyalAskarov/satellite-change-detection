from pathlib import Path
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
from model import SiameseUNet, image_tensor, ARCH, MEAN, STD

def validate_dataset(root):
    root = Path(root)
    expected = {'train':445, 'val':64, 'test':128}
    manifest = {}
    for split,n in expected.items():
        groups = [{p.name for p in (root/split/d).glob('*.png')} for d in ['A','B','label']]
        if not groups[0] or not groups[0] == groups[1] == groups[2]:
            raise ValueError(f'{split}: A, B and label PNG filenames must match exactly.')
        if len(groups[0]) != n:
            raise ValueError(f'{split}: expected original LEVIR-CD {n} pairs, found {len(groups[0])}. Check the download; do not mix LEVIR-CD+ or prepatched data.')
        for name in tqdm(sorted(groups[0]), desc=f'Checking {split}'):
            for folder in ['A','B','label']:
                with Image.open(root/split/folder/name) as im:
                    if im.size != (1024,1024):
                        raise ValueError(f'Unexpected size: {split}/{folder}/{name}: {im.size}')
                    if folder == 'label':
                        values = set(np.unique(np.asarray(im.convert('L'))).tolist())
                        if not values.issubset({0,1,255}):
                            raise ValueError(f'Nonbinary mask {name}: {values}')
                    else:
                        im.verify()
        manifest[split] = sorted(groups[0])
        print(split, len(groups[0]), 'verified pairs')
    return manifest

class LEVIRPatches(Dataset):
    def __init__(self, root, split, tile=256, augment=False):
        self.root, self.split, self.tile, self.augment = Path(root), split, tile, augment
        if 1024 % tile:
            raise ValueError('Tile must divide the 1024-pixel scenes.')
        self.names = sorted(p.name for p in (self.root/split/'A').glob('*.png'))
        self.grid = 1024//tile
        self.cached_name = None
        self.cached_arrays = None

    def __len__(self):
        return len(self.names)*self.grid*self.grid

    def __getitem__(self, index):
        scene, patch = divmod(index,self.grid*self.grid)
        name = self.names[scene]
        # One-scene cache accelerates sequential validation without loading all data into RAM.
        if name != self.cached_name:
            arrays = []
            for folder in ['A','B','label']:
                with Image.open(self.root/self.split/folder/name) as im:
                    arrays.append(np.array(im.convert('L' if folder=='label' else 'RGB')))
            self.cached_name, self.cached_arrays = name, arrays
        y,x = (patch//self.grid)*self.tile, (patch%self.grid)*self.tile
        a,b,m = [v[y:y+self.tile,x:x+self.tile].copy() for v in self.cached_arrays]
        if self.augment:
            k = random.randrange(4)
            a,b,m = [np.rot90(v,k).copy() for v in (a,b,m)]
            if random.random() < .5:
                a,b,m = [np.fliplr(v).copy() for v in (a,b,m)]
            if random.random() < .5:
                a,b = b,a
        return image_tensor(a),image_tensor(b),torch.from_numpy((m>0).astype(np.float32))[None]

def loss_fn(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits,target)
    p = logits.sigmoid()
    dice = (2*(p*target).sum((1,2,3))+1)/(p.sum((1,2,3))+target.sum((1,2,3))+1)
    return .5*bce+.5*(1-dice.mean())

def metrics(tp,fp,fn,tn):
    # Global changed-class metrics, not averages of mostly empty patch scores.
    return {'precision':tp/max(tp+fp,1), 'recall':tp/max(tp+fn,1),
            'f1':2*tp/max(2*tp+fp+fn,1), 'iou':tp/max(tp+fp+fn,1),
            'accuracy':(tp+tn)/max(tp+fp+fn+tn,1), 'tp':tp,'fp':fp,'fn':fn,'tn':tn}

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    counts = torch.zeros(4, dtype=torch.int64, device=device)
    for a,b,y in tqdm(loader,desc='Evaluate',leave=False):
        a,b,y = a.to(device),b.to(device),y.to(device).bool()
        p = model(a,b).sigmoid() >= .5
        counts += torch.stack([(p&y).sum(), (p&~y).sum(), (~p&y).sum(), (~p&~y).sum()])
    return metrics(*counts.cpu().tolist())

def atomic_save(value,path):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(value,temp)
    temp.replace(path)

def train(root, run_dir, epochs=30, batch_size=8, workers=2, resume=True):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        raise RuntimeError('Select a GPU runtime in Colab, reconnect, and rerun the notebook.')
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True,exist_ok=True)
    last = run_dir/'last.pt'
    continuing = resume and last.exists()
    if not continuing and (last.exists() or (run_dir/'best.pt').exists()):
        raise ValueError('Choose an empty RUN_DIR to start a new experiment.')
    model = SiameseUNet(pretrained=not continuing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    best,start,stale,history = -1.,0,0,[]
    if continuing:
        c = torch.load(last,map_location='cpu',weights_only=True)
        if c['architecture'] != ARCH or c['tile_size'] != 256:
            raise ValueError('Incompatible training checkpoint.')
        if not (run_dir/'best.pt').exists():
            raise ValueError('Resume needs both last.pt and best.pt in RUN_DIR.')
        model.load_state_dict(c['model_state'])
        optimizer.load_state_dict(c['optimizer'])
        scaler.load_state_dict(c['scaler'])
        best,start,stale,history = c['best_f1'],c['epoch']+1,c['stale'],c['history']
        print('Resuming at epoch',start+1,'(data order may differ after reconnect)')
    train_loader = DataLoader(LEVIRPatches(root,'train',augment=True),batch_size=batch_size,
                              shuffle=True,num_workers=workers,pin_memory=True)
    val_loader = DataLoader(LEVIRPatches(root,'val'),batch_size=batch_size,
                            num_workers=workers,pin_memory=True)
    for epoch in range(start,epochs):
        model.train()
        total = 0.
        seen = 0
        for a,b,y in tqdm(train_loader,desc=f'Epoch {epoch+1}/{epochs}'):
            a,b,y = a.to(device),b.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                loss = loss_fn(model(a,b),y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss; stop and inspect the data.')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()*len(y)
            seen += len(y)
        val = evaluate(model,val_loader,device)
        history.append({'epoch':epoch+1,'train_loss':total/seen,**{'val_'+k:v for k,v in val.items()}})
        improved = val['f1'] > best
        stale = 0 if improved else stale+1
        best = max(best,val['f1'])
        c = {'architecture':ARCH,'model_state':model.state_dict(),'mean':list(MEAN),'std':list(STD),
             'tile_size':256,'threshold':.5,'epoch':epoch,'best_f1':best,'stale':stale,
             'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'history':history,
             'torch_version':str(torch.__version__)}
        if improved:
            atomic_save(c,run_dir/'best.pt')
        atomic_save(c,last)
        (run_dir/'history.json').write_text(json.dumps(history,indent=2))
        print(f"Loss {total/seen:.4f} | Val F1 {val['f1']:.4f} | IoU {val['iou']:.4f} | best F1 {best:.4f}")
        if stale >= 8:
            print('Early stopping: no validation F1 improvement for 8 epochs.')
            break
    return history
