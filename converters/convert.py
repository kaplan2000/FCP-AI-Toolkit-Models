#!/usr/bin/env python3
"""Portable prototype Core ML conversion recipe; not an application runtime.

New wrapper: MIT, see LICENSES/CONVERTERS-MIT.txt. Vendored source retains its
original licenses. No model import occurs without --execute. Rebuilding is not
bit-identical package reproduction or a claim that the result is qualified.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def source_module(name, relative, changes=()):
    path = ROOT / relative
    code = path.read_text()
    for old, new in changes:
        if code.count(old) != 1:
            raise ValueError('Unexpected source revision: ' + relative)
        code = code.replace(old, new)
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(code, relative, 'exec'), module.__dict__)
    return module

def build(kind, checkpoint, torch):
    if kind == 'x4plus':
        from torch import nn
        from torch.nn import init
        from torch.nn.modules.batchnorm import _BatchNorm
        tree = ast.parse((ROOT/'vendor/basicsr/arch_util.py').read_text())
        names = {'default_init_weights', 'make_layer', 'pixel_unshuffle'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes} == names
        scope = {'torch':torch, 'nn':nn, 'init':init, '_BatchNorm':_BatchNorm}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'vendor/basicsr/arch_util.py', 'exec'), scope)
        code = (ROOT/'vendor/basicsr/rrdbnet_arch.py').read_text()
        for old in ('from basicsr.utils.registry import ARCH_REGISTRY\n', 'from .arch_util import default_init_weights, make_layer, pixel_unshuffle\n', '@ARCH_REGISTRY.register()\n'):
            assert old in code
            code = code.replace(old, '')
        exec(compile(code, 'vendor/basicsr/rrdbnet_arch.py', 'exec'), scope)
        model = scope['RRDBNet'](num_in_ch=3,num_out_ch=3,num_feat=64,num_block=23,num_grow_ch=32,scale=4)
        state = torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(state.get('params_ema',state.get('params',state)),strict=True)
    elif kind == 'instructir':
        sys.modules['models'] = types.ModuleType('models')
        for name in ('nafnet_utils','nafnet','instructir'):
            source_module('models.'+name, 'vendor/instructir/models/'+name+'.py')
        def layernorm(self,x):
            mean=x.mean(1,keepdim=True)
            variance=(x-mean).pow(2).mean(1,keepdim=True)
            return self.weight.view(1,-1,1,1)*((x-mean)/(variance+self.eps).sqrt())+self.bias.view(1,-1,1,1)
        sys.modules['models.nafnet_utils'].LayerNorm2d.forward=layernorm
        model=sys.modules['models.instructir'].create_model(input_channels=3,width=32,enc_blks=[2,2,4,8],middle_blk_num=4,dec_blks=[2,2,2,2],txtdim=256)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        model.check_image_size=types.MethodType(lambda self,x:x,model)
    elif kind == 'rife':
        sys.modules['model']=types.ModuleType('model')
        source_module('model.warplayer','vendor/rife425/warplayer.py')
        module=source_module('rife_conversion','vendor/rife425/IFNet_HDv3.py')
        network=module.IFNet()
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        state={k.removeprefix('module.'):v for k,v in state.items()}
        state={k:v for k,v in state.items() if not k.startswith(('teacher.','caltime.'))}
        network.load_state_dict(state,strict=True)
        class RIFEInference(torch.nn.Module):
            def __init__(self):
                super().__init__();self.network=network
            def forward(self,images,timestep):
                return self.network(images,timestep,[16,8,4,2,1])[2][-1]
        model=RIFEInference()
    else:
        module=source_module('fbcnn_conversion','vendor/fbcnn/network_fbcnn.py', [('import torchvision.models as models\n','')])
        model=module.FBCNN(in_nc=3,out_nc=3,nc=[64,128,256,512],nb=4,act_mode='R')
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    return model.cpu().eval()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',choices=['x4plus','instructir','rife','fbcnn'])
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New .mlpackage destination; never overwritten')
    parser.add_argument('--sample-npy',type=Path,help='Optional user-owned float32 NCHW trace sample; fixed shape for variant. Never copied into the package.')
    parser.add_argument('--embeddings',type=Path,default=ROOT/'provenance/fixed-task-embeddings.json')
    parser.add_argument('--execute',action='store_true',help='Run model loading, tracing and conversion; absent means hash checks only')
    args=parser.parse_args()
    catalog=json.loads((ROOT/'provenance/models.json').read_text())
    spec=next(x for x in catalog['models'] if x['recipeKey']==args.model)
    if sha(args.checkpoint)!=spec['checkpoint']['sha256']:
        parser.error('Checkpoint SHA-256 differs from the pinned official input')
    for row in json.loads((ROOT/'provenance/upstream-files.json').read_text())['files']:
        if sha(ROOT/row['path'])!=row['sha256']:parser.error('Vendored source/license identity mismatch')
    if args.model=='instructir' and sha(args.embeddings)!=catalog['fixedTaskVectors']['sha256']:
        parser.error('Frozen task vector file identity mismatch')
    if args.output.suffix!='.mlpackage' or args.output.exists():parser.error('Use a new .mlpackage path')
    if not args.execute:
        print('Pinned input identities pass. No model imported; --execute is required for conversion.')
        return
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
    import numpy as np
    import torch
    import coremltools as ct
    if torch.cuda.is_available():raise RuntimeError('This frozen recipe targets Apple/CPU tracing, not CUDA')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(1202)
    model=build(args.model,args.checkpoint,torch)
    shape={'x4plus':(1,3,256,256),'instructir':(1,3,720,1280),'rife':(1,6,512,512),'fbcnn':(1,3,512,512)}[args.model]
    if args.sample_npy:
        a=np.load(args.sample_npy,allow_pickle=False)
        if a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all() or a.min()<0 or a.max()>1:raise ValueError('Expected finite float32 encoded RGB sample in [0,1] at exact variant shape')
        sample=torch.from_numpy(np.ascontiguousarray(a))
    else:sample=torch.rand(shape)
    inputs=[ct.TensorType(name='input',shape=shape,dtype=np.float32)]
    samples=(sample,)
    if args.model=='instructir':
        vectors=json.loads(args.embeddings.read_text())['tasks']
        vector=torch.tensor(vectors['denoise']['embedding'],dtype=torch.float32)
        samples=(sample,vector)
        inputs=[ct.TensorType(name='input',shape=(1,3,ct.RangeDim(16,3840,default=720),ct.RangeDim(16,3840,default=1280)),dtype=np.float32),ct.TensorType(name='text_embedding',shape=(1,256),dtype=np.float32)]
    elif args.model=='rife':
        samples=(sample,torch.tensor([[[[.5]]]],dtype=torch.float32))
        inputs.append(ct.TensorType(name='timestep',shape=(1,1,1,1),dtype=np.float32))
    with torch.no_grad():traced=torch.jit.trace(model,samples,check_trace=args.model!='fbcnn')
    outputs=[ct.TensorType(name='output',dtype=np.float32)]
    if args.model=='fbcnn':outputs.append(ct.TensorType(name='qf',dtype=np.float32))
    converted=ct.convert(traced,convert_to='mlprogram',inputs=inputs,outputs=outputs,
                         compute_precision=ct.precision.FLOAT16 if args.model=='x4plus' else ct.precision.FLOAT32,
                         minimum_deployment_target=ct.target.macOS13,compute_units=ct.ComputeUnit.CPU_AND_GPU,skip_model_load=True)
    metadata=spec['originalPackageMetadata']
    converted.short_description=metadata.get('shortDescription','')
    converted.author=metadata.get('author','')
    converted.license=metadata.get('license','')
    converted.user_defined_metadata.update(metadata.get('userDefined',{}))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    converted.save(str(args.output))
    print('Converted package saved. Numerical, native GPU, capacity and quality validation are still required for regenerated bytes.')

if __name__=='__main__':main()
