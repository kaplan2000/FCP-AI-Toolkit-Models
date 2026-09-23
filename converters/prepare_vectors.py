#!/usr/bin/env python3
"""Offline regeneration of the two public InstructIR task vectors.
New wrapper: MIT, see LICENSES/CONVERTERS-MIT.txt. Upstream helpers retain MIT.
Requires explicit local inputs. Does not download files or inspect images.
"""
import argparse, ast, hashlib, json, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    mp=ROOT/'provenance/instructir-upstream-inputs.json';manifest=json.loads(mp.read_text())
    for row in manifest['inputs']:
        path=args.inputs_dir/row['name']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:p.error('Official input identity mismatch: '+row['name'])
    if args.output.exists():p.error('Output exists; preserve prior evidence')
    if not args.execute:
        print('Input identities pass; --execute required for CPU vector generation.');return
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[name]='2'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch,transformers
    from torch import nn
    import torch.nn.functional as F
    from transformers import AutoModel,AutoTokenizer
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    source=ROOT/'vendor/instructir/text/models.py'
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in {'mean_pooling','LMHead'}]
    scope={'torch':torch,'nn':nn,'F':F}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'vendor/instructir/text/models.py','exec'),scope)
    tokenizer=AutoTokenizer.from_pretrained(str(args.inputs_dir/'bge-micro-v2'),local_files_only=True)
    model=AutoModel.from_pretrained(str(args.inputs_dir/'bge-micro-v2'),local_files_only=True).cpu().eval()
    head=scope['LMHead'](embedding_dim=384,hidden_dim=256,num_classes=7)
    head.load_state_dict(torch.load(args.inputs_dir/'lm_instructir-7d.pt',map_location='cpu',weights_only=True),strict=True);head.eval()
    prompts={'denoise':'I need this image denoised ASAP.','deblur':'Please, clean up this blurry photo.'}
    values={}
    for task,prompt in prompts.items():
        tokens=tokenizer(prompt,padding=True,truncation=True,return_tensors='pt')
        if tokens['input_ids'].shape[1]>512:raise ValueError('Prompt exceeds position capacity')
        with torch.no_grad():
            pooled=scope['mean_pooling'](model(**tokens),tokens['attention_mask'])
            embedding,_=head(F.normalize(pooled,p=2,dim=1))
        values[task]={'prompt':prompt,'tokens':{k:v.tolist() for k,v in tokens.items()},'embedding':embedding.tolist(),'shape':[1,256],'float32SHA256':hashlib.sha256(embedding.numpy().astype('<f4').tobytes()).hexdigest()}
    result={'schemaVersion':1,'promptsSourceRevision':manifest['sourceRevision'],'imageCheckpointRevision':manifest['imageCheckpointRevision'],'languageRevision':manifest['languageRevision'],'inputsManifestSHA256':sha(mp),'torch':torch.__version__,'transformers':transformers.__version__,'device':'cpu','threads':2,'semantics':'Official fixed task prompts; one shared restoration checkpoint; combined noise+blur not qualified','tasks':values}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    expected=json.loads((ROOT/'provenance/fixed-task-embeddings.json').read_text())['tasks']
    if any(values[k]['float32SHA256']!=expected[k]['float32SHA256'] for k in values):raise RuntimeError('Regenerated vectors differ from published vector identities; do not substitute silently')
    print('Both regenerated raw float32 task-vector hashes match.')
if __name__=='__main__':main()
