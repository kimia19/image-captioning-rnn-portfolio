"""Minimal real-COCO-2017 captioning rehearsal (NOT the original COCO-2014 assignment).

The COCO-2017 validation annotations are checksum-pinned, and individual
images are obtained from a reference COCO mirror. No starter code is used. Only six training photos and two validation photos are used.
This exercises an independently written vanilla-RNN captioner on real
images with real caption tokens; it is NOT a caption-quality benchmark.

Dataset official: https://cocodataset.org/
Mirror: https://huggingface.co/datasets/merve/coco/blob/main/annotations/captions_val2017.json
Per-photo license is recorded; only public-license IDs 4,5,7,8 are selected.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import re
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor, Resize, Compose

from captioning_model import CaptioningRNN, decode_captions, train_captioner

CAPTIONS="https://huggingface.co/datasets/merve/coco/resolve/main/annotations/captions_val2017.json"
CAPTIONS_SHA256="afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9"
IMAGE_ROOT="https://huggingface.co/datasets/merve/coco/resolve/main/val2017/"
PERMITTED_COCO_LICENSE_IDS={4,5,7,8}


def fetch(url):
    request=Request(url,headers={"User-Agent":"Academic-reproducibility/1.0"})
    with urlopen(request,timeout=45) as response:
        return response.read()


def load_real_subset(num_images=8):
    if not 8 <= num_images <= 30:
        raise ValueError("Tiny subset only (8-30)")
    raw=fetch(CAPTIONS)
    received=hashlib.sha256(raw).hexdigest()
    if received != CAPTIONS_SHA256:
        raise ValueError(f"Annotation SHA256 differs from pinned source: {received}")
    ann=json.loads(raw)
    captions=defaultdict(list)
    for item in ann["annotations"]:
        captions[int(item["image_id"])].append(item["caption"])
    transform=Compose([Resize((112,112)),ToTensor()])
    result=[]
    attempted=0
    # Deterministic, no selection by labels/caption content.
    for meta in sorted(ann["images"],key=lambda x:x["id"]):
        if int(meta.get("license",-1)) not in PERMITTED_COCO_LICENSE_IDS:
            continue
        if not captions[int(meta["id"])]:
            continue
        filename=str(meta["file_name"])
        if not re.fullmatch(r"\d{12}\.jpg",filename):
            continue
        attempted+=1
        if attempted>60:
            break
        try:
            data=fetch(IMAGE_ROOT+filename)
            image=Image.open(io.BytesIO(data)).convert("RGB")
            image.load()
            result.append({
                "id":int(meta["id"]),"file_name":filename,
                "sha256":hashlib.sha256(data).hexdigest(),
                "license_id":int(meta["license"]),
                "image":transform(image),
                "caption":captions[int(meta["id"])][0]
            })
        except Exception as exc:
            print(f"Skipped unavailable COCO image ID {meta['id']}: {type(exc).__name__}",flush=True)
        if len(result)==num_images:
            break
    if len(result)<num_images:
        raise RuntimeError(f"Downloaded {len(result)} / {num_images} real COCO photos after {attempted} attempts")
    return result


def vocabulary_and_tensors(rows,train_size=6,max_tokens=13):
    tokenized=[
        re.findall(r"[a-z]+",row["caption"].lower())
        for row in rows
    ]
    counts=Counter(t for tokens in tokenized[:train_size] for t in tokens)
    vocab={"<NULL>":0,"<START>":1,"<END>":2,"<UNK>":3}
    for token,_ in sorted(counts.items(),key=lambda kv:(-kv[1],kv[0])):
        vocab[token]=len(vocab)
    captions=torch.zeros((len(rows),max_tokens),dtype=torch.long)
    for i,tokens in enumerate(tokenized):
        encoded=[vocab["<START>"]]
        encoded.extend(vocab.get(t,vocab["<UNK>"]) for t in tokens[:max_tokens-2])
        encoded.append(vocab["<END>"])
        captions[i,:len(encoded)]=torch.tensor(encoded)
    photos=torch.stack([r["image"] for r in rows])
    return vocab,photos,captions


def run(output,epochs=2,seed=42):
    if not 1<=epochs<=3:
        raise ValueError("Only 1-3 epochs supported by this tiny pilot")
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(),2))
    rows=load_real_subset(8)
    vocab,photos,caps=vocabulary_and_tensors(rows,train_size=6)
    # Pretrained RegNet ImageNet weights can be downloaded by torchvision.
    # To keep the tiny CPU rehearsal predictable, freeze the backbone.
    model=CaptioningRNN(
        vocab,input_dim=400,wordvec_dim=32,hidden_dim=32,
        cell_type="rnn",image_encoder_pretrained=True
    )
    model.image_encoder.requires_grad_(False)
    model,history=train_captioner(
        model,photos[:6],caps[:6],num_epochs=epochs,
        batch_size=2,learning_rate=1e-3,device="cpu"
    )
    model.eval()
    with torch.no_grad():
        validation_loss=float(model(photos[6:],caps[6:]))
        predicted=model.sample(photos[6:],max_length=12)
    predictions=decode_captions(predicted,{i:w for w,i in vocab.items()})
    report={
        "experiment":"Eight real COCO-2017 photos, tiny vanilla-RNN caption training rehearsal",
        "not_original_assignment_coco2014_reproduction":True,
        "annotation_source":CAPTIONS,
        "image_mirror":"Merve COCO dataset (byte-preserving COCO val2017 mirror); each image SHA256 recorded",
        "annotation_sha256":CAPTIONS_SHA256,
        "image_source":IMAGE_ROOT,
        "permitted_license_ids":sorted(PERMITTED_COCO_LICENSE_IDS),
        "selected_images":[{
            "id":x["id"],"file_name":x["file_name"],
            "sha256":x["sha256"],"license_id":x["license_id"]
        }for x in rows],
        "train_images":6,"validation_images":2,
        "seed":seed,"epochs":epochs,"batch_size":2,
        "architecture":"Independent captioning_model.py vanilla RNN and frozen ImageNet pretrained RegNet; 32D word/hidden state",
        "vocabulary_size":len(vocab),
        "training_mean_loss_by_epoch":history,
        "validation_loss":validation_loss,
        "sample_ids":[r["id"] for r in rows[6:]],
        "generated_captions":predictions,
        "limitations":[
            "Only 6 train images: inadequate to learn a meaningful image-to-text map",
            "Image source is COCO-2017 validation, not original course COCO-2014 dataset",
            "Greedy generated outputs are diagnostic, not evidence of quality",
            "No BLEU/CIDEr or benchmark evaluation claimed",
            "Individual image licenses and attribution must be respected; no photos committed"
        ],
        "torch_version":torch.__version__
    }
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("Validation loss:",validation_loss,flush=True)
    print("Generated diagnostic captions:",predictions,flush=True)
    return report


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--output",type=Path,default=Path("outputs/coco2017_mini_pilot.json"))
    a.add_argument("--epochs",type=int,default=2)
    z=a.parse_args();run(z.output,z.epochs)
