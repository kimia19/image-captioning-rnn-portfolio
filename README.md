# Image Captioning with a Vanilla RNN

An independent portfolio implementation of an **ImageNet-pretrained RegNet visual encoder** and **vanilla RNN decoder** for generating short image descriptions. This edition was independently authored rather than copying instructional notebook scaffolding, and contains no historical coursework commits.

## Inputs

The demonstration uses a deterministic, very small sample of the **COCO-2017 validation** captions and photographs. COCO annotations: https://cocodataset.org/ ; reference mirror: https://huggingface.co/datasets/merve/coco. The individual photographs retain owner rights; only selected permitted Flickr/COCO license IDs are used, and image hashes are recorded. No photos or pretrained model checkpoints are committed.

## Run

```bash
python -m pip install -r requirements.txt
python real_coco_demo.py --epochs 2
```

The script obtains 8 real images and captions, trains on only six, and reports validation loss / greedy captions on two other images. It is a software integration demonstration, **not a credible caption-quality benchmark**. A pretrained Torchvision RegNet is downloaded automatically; network connectivity is required.

## Technical scope

The encoder extracts convolutional image features; mean-pooled features initialize the recurrent hidden state. Token embeddings feed a PyTorch vanilla RNN, which predicts the next token through cross-entropy and uses greedy sampling for demonstration. **LSTM and attention are intentionally not implemented.**

## Research limitations

Six training images cannot establish generalization. Outputs may be repetitive or semantically incorrect; BLEU, CIDEr and other published COCO metrics are not reported. The original educational exercise used a separate preprocessed COCO-2014 file, not distributed here. No explicit code license has been granted pending an owner decision.
