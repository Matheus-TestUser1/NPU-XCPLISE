#!/usr/bin/env python3
"""Train a tiny RGB 2x ESPCN-style demo and export NNAPI weights as FP16/base64.

Dependencies: pip install torch numpy pillow scikit-image
Uses bundled scikit-image examples for research. No photos are redistributed.
Run: python tools/train_espcn.py --steps 700
"""
import argparse
import base64
import math
import random
import struct
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage import data

FEATURES=12

class TinyESPCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,FEATURES,3,padding=1)
        self.conv2=nn.Conv2d(FEATURES,12,3,padding=1)
        with torch.no_grad():
            self.conv1.weight.zero_()
            self.conv1.bias.zero_()
            for channel in range(3):
                self.conv1.weight[channel,channel,1,1]=1.
            self.conv2.weight.zero_()
            self.conv2.bias.zero_()
            nn.init.normal_(self.conv1.weight[3:],std=.025)
            nn.init.normal_(self.conv2.weight[:,3:],std=.003)
            for block in range(4):
                for channel in range(3):
                    self.conv2.weight[block*3+channel,channel,1,1]=1.

    def forward(self,x):
        z=self.conv2(F.relu(self.conv1(x)))
        b,c,h,w=z.shape
        # NNAPI DEPTH_TO_SPACE (CRD): 4 blocks x 3 RGB channels.
        z=z.view(b,4,3,h,w).permute(0,2,1,3,4).reshape(b,12,h,w)
        return F.pixel_shuffle(z,2)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=700)
    parser.add_argument('--seed',type=int,default=4)
    parser.add_argument('--output',type=Path,default=Path('app/src/main/assets/espcn2x_fp16.b64'))
    args=parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    images=[]
    for name in ('astronaut','coffee','chelsea','cat','colorwheel','brick','grass','gravel'):
        arr=getattr(data,name)()
        if arr.ndim==2:
            arr=np.repeat(arr[...,None],3,axis=2)
        if arr.dtype!=np.uint8:
            arr=(np.clip(arr,0,1)*255).astype('uint8')
        h,w=arr.shape[:2]
        h-=h%2
        w-=w%2
        hi=Image.fromarray(arr[:h,:w])
        lo=hi.resize((w//2,h//2),Image.Resampling.BICUBIC)
        images.append((np.asarray(lo),np.asarray(hi)))

    def batch(n=8,patch=40):
        xs=[]
        ys=[]
        for _ in range(n):
            lo,hi=random.choice(images)
            h,w=lo.shape[:2]
            x=random.randrange(0,w-patch)
            y=random.randrange(0,h-patch)
            a=np.array(lo[y:y+patch,x:x+patch],copy=True)
            b=np.array(hi[2*y:2*(y+patch),2*x:2*(x+patch)],copy=True)
            if random.random()<.5:
                a=a[:,::-1].copy()
                b=b[:,::-1].copy()
            xs.append(torch.from_numpy(a.copy()).permute(2,0,1).float()/255)
            ys.append(torch.from_numpy(b.copy()).permute(2,0,1).float()/255)
        return torch.stack(xs),torch.stack(ys)

    model=TinyESPCN()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    for i in range(args.steps):
        x,y=batch()
        optimizer.zero_grad(set_to_none=True)
        loss=F.mse_loss(model(x),y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if i%100==0 or i==args.steps-1:
            print('step',i,'mse',float(loss.detach()))

    with torch.no_grad():
        reference=np.array(data.rocket(),copy=True)
        h,w=reference.shape[:2]
        h-=h%2
        w-=w%2
        reference=reference[:h,:w]
        low=np.array(Image.fromarray(reference).resize((w//2,h//2),Image.Resampling.BICUBIC))
        x=torch.from_numpy(low.copy()).permute(2,0,1).unsqueeze(0).float()/255
        ref=torch.from_numpy(reference.copy()).permute(2,0,1).unsqueeze(0).float()/255
        pred=model(x).clamp(0,1)
        bicubic=F.interpolate(x,scale_factor=2,mode='bicubic',align_corners=False).clamp(0,1)
        def psnr(t):
            return -10*math.log10(max(float(F.mse_loss(t,ref)),1e-12))
        print('HELDOUT rocket PSNR neural / bicubic',psnr(pred),psnr(bicubic))

    def export_filter(conv):
        # torch OIHW to NNAPI OHWI; little-endian half-float
        return conv.weight.detach().permute(0,2,3,1).contiguous().numpy().astype('float32').ravel()
    arrays=[
        export_filter(model.conv1),model.conv1.bias.detach().numpy().astype('float32'),
        export_filter(model.conv2),model.conv2.bias.detach().numpy().astype('float32')
    ]
    binary=b'ESPCN2X1'+struct.pack('<5I',FEATURES,*[len(a) for a in arrays])
    binary+=b''.join(struct.pack('<e',float(v)) for array in arrays for v in array)
    encoded=base64.b64encode(binary).decode('ascii')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(encoded[i:i+96] for i in range(0,len(encoded),96))+'\n')
    print('saved',args.output,'bytes',len(binary))

if __name__=='__main__':
    main()
