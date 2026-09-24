#!/usr/bin/env python3
"""Pinned animevideov3 with the installed NCNN x2 graph's bicubic reduction.
Developer conversion only. Writes new artifacts; never overwrites old packages.
"""
import argparse, hashlib, json, struct, time
from pathlib import Path
import numpy as np
import torch
import coremltools as ct

BASE = Path(__file__).resolve().parents[1]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--scale',type=int,choices=[2,4],default=2);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--ncnn-weights',type=Path,required=True);args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    source=BASE/'vendor/realesrgan/srvgg_arch.py'
    ns={};exec(compile(source.read_text().replace('from basicsr.utils.registry import ARCH_REGISTRY\n','').replace('@ARCH_REGISTRY.register()\n',''),str(source),'exec'),ns)
    net=ns['SRVGGNetCompact'](num_in_ch=3,num_out_ch=3,num_feat=64,num_conv=16,upscale=4,act_type='prelu').eval()
    checkpoint=args.checkpoint
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);net.load_state_dict(state.get('params_ema',state.get('params',state)),strict=True)
    # Verify the installed NCNN file contains these same official weights.
    binary=args.ncnn_weights;buf=binary.read_bytes();offset=0;checks=[]
    for i,layer in enumerate(net.body):
        for name,param in layer.named_parameters():
            expected=param.detach().numpy().reshape(-1)
            if isinstance(layer,torch.nn.Conv2d) and name=='weight':
                tag=struct.unpack_from('<I',buf,offset)[0];offset+=4;assert tag==0x01306B47,hex(tag)
                observed=np.frombuffer(buf,dtype='<f2',count=expected.size,offset=offset);offset+=(expected.size*2+3)//4*4
                rounded=expected.astype(np.float16)
                truncated=np.where(np.abs(rounded.astype(np.float32))>np.abs(expected),np.nextafter(rounded,np.float16(0)),rounded)
                truncated=np.where(np.abs(truncated)<np.finfo(np.float16).tiny,np.copysign(np.float16(0),truncated),truncated)
                assert np.array_equal(observed,truncated),(i,name)
                # NCNN's fp16 serializer truncates rather than round-to-nearest.
                # Use its exact stored weights in both engines.
                with torch.no_grad(): param.copy_(torch.from_numpy(observed.astype(np.float32).reshape(param.shape)))
            else:
                observed=np.frombuffer(buf,dtype='<f4',count=expected.size,offset=offset);offset+=expected.size*4
                assert np.array_equal(observed,expected),(i,name)
            checks.append(f'body.{i}.{name}')
    assert offset==len(buf),(offset,len(buf))
    class X2(torch.nn.Module):
        def __init__(self):
            super().__init__();self.net=net
            k=torch.tensor([-.09375,.59375,.59375,-.09375]);self.register_buffer('kernel',(k[:,None]*k[None,:])[None,None].repeat(3,1,1,1))
        def forward(self,x):
            y=self.net(x)
            return torch.nn.functional.conv2d(torch.nn.functional.pad(y,(1,1,1,1),mode='replicate'),self.kernel,stride=2,groups=3)
    torch.set_num_threads(2);torch.manual_seed(1202)
    x=torch.rand(1,3,256,256);wrapped=X2().eval() if args.scale==2 else net
    with torch.no_grad():
        ref=wrapped(x);bicubic=torch.nn.functional.interpolate(net(x),scale_factor=.5,mode='bicubic',align_corners=False) if args.scale==2 else net(x)
        resize_error=float((ref-bicubic).abs().max());assert resize_error<2e-6,resize_error
        traced=torch.jit.trace(wrapped,x)
    started=time.monotonic()
    model=ct.convert(traced,convert_to='mlprogram',inputs=[ct.TensorType(name='input',shape=x.shape,dtype=np.float32)],outputs=[ct.TensorType(name='output',dtype=np.float32)],compute_precision=ct.precision.FLOAT16,minimum_deployment_target=ct.target.macOS13,compute_units=ct.ComputeUnit.CPU_AND_GPU)
    package=out/f'realesr-animevideov3-x{args.scale}-tile256-fp16.mlpackage';assert not package.exists()
    model.short_description=f'Real-ESRGAN animevideov3, NCNN weights, {args.scale}x output; fixed tile256.'
    model.user_defined_metadata['checkpoint_sha256']=sha(checkpoint)
    model.user_defined_metadata['ncnn_weights_sha256']=sha(binary)
    model.save(str(package));convert_seconds=time.monotonic()-started
    started=time.monotonic();pred=np.asarray(model.predict({'input':x.numpy()})['output']);elapsed=time.monotonic()-started
    diff=np.abs(pred-ref.numpy());assert pred.shape==(1,3,256*args.scale,256*args.scale) and np.isfinite(pred).all()
    assert diff.mean()<.002 and np.quantile(diff,.99)<.01
    result={'status':'pass','checkpointSHA256':sha(checkpoint),'ncnnWeightsSHA256':sha(binary),'officialSourceSHA256':sha(source),'verifiedWeightTensors':len(checks),'ncnnWeightMatch':'all convolution weights match fp16 truncation/denormal flush of official checkpoint; fp32 biases and slopes exact; Core ML uses exact NCNN-stored convolution weights','resize':'none; native 4x' if args.scale==4 else 'bicubic half-pixel 4x to 2x','resizeMaxError':resize_error,'conversionSeconds':convert_seconds,'firstPredictionSeconds':elapsed,'coremlTorchParity':{'meanAbsoluteError':float(diff.mean()),'p99':float(np.quantile(diff,.99)),'max':float(diff.max())},'package':str(package),'input':[1,3,256,256],'output':[1,3,256*args.scale,256*args.scale],'files':[{'path':str(p.relative_to(package)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(package.rglob('*')) if p.is_file()]}
    (out/'conversion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
if __name__=='__main__':main()
