import numpy as np
import torch
from model import image_tensor

def positions(length, tile, stride):
    if length <= tile:
        return [0]
    return sorted(set(list(range(0, length-tile+1, stride)) + [length-tile]))

@torch.inference_mode()
def predict_pair(model, before, after, device='cpu', tile=256, stride=256, progress=None):
    a, b = np.asarray(before), np.asarray(after)
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError('Images must be RGB with matching dimensions and aligned pixels.')
    if not 0 < stride <= tile:
        raise ValueError('Stride must be between 1 and tile size.')
    h, w = a.shape[:2]
    pad = ((0,max(0,tile-h)), (0,max(0,tile-w)), (0,0))
    a, b = np.pad(a,pad,mode='edge'), np.pad(b,pad,mode='edge')
    total = np.zeros(a.shape[:2], dtype=np.float32)
    count = np.zeros_like(total)
    coords = [(y,x) for y in positions(a.shape[0],tile,stride) for x in positions(a.shape[1],tile,stride)]
    model.eval()
    for i,(y,x) in enumerate(coords):
        ta = image_tensor(a[y:y+tile,x:x+tile]).unsqueeze(0).to(device)
        tb = image_tensor(b[y:y+tile,x:x+tile]).unsqueeze(0).to(device)
        p = model(ta,tb).sigmoid()[0,0].cpu().numpy()
        total[y:y+tile,x:x+tile] += p
        count[y:y+tile,x:x+tile] += 1
        if progress:
            progress((i+1)/len(coords))
    return (total/count)[:h,:w]
