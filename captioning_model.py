"""Independently authored RegNet + vanilla RNN image captioner.

Educational code for a fresh public portfolio repository. No course starter
text or pretrained model weights are bundled. Uses PyTorch and torchvision.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import regnet_x_400mf, RegNet_X_400MF_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class VisualEncoder(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        backbone=regnet_x_400mf(
            weights=RegNet_X_400MF_Weights.DEFAULT if pretrained else None
        )
        self.cnn=create_feature_extractor(
            backbone,return_nodes={"trunk_output.block4":"features"}
        )
        self.normalizer=nn.Identity()
        self.register_buffer("mean",torch.tensor([.485,.456,.406]).reshape(1,3,1,1))
        self.register_buffer("std",torch.tensor([.229,.224,.225]).reshape(1,3,1,1))
        self.out_channels=400

    def forward(self,photos):
        x=photos.float() / (255. if photos.dtype==torch.uint8 else 1.)
        return self.cnn((x-self.mean)/self.std)["features"]


class CaptioningRNN(nn.Module):
    def __init__(self,word_to_idx,input_dim=400,wordvec_dim=32,
                 hidden_dim=32,cell_type="rnn",
                 image_encoder_pretrained=True):
        super().__init__()
        if cell_type!="rnn":
            raise NotImplementedError("Portfolio edition supports vanilla RNN only")
        if not {"<START>","<END>","<NULL>"}.issubset(word_to_idx):
            raise ValueError("Required special caption tokens missing")
        self.word_to_idx=word_to_idx
        self.idx_to_word={v:k for k,v in word_to_idx.items()}
        self.image_encoder=VisualEncoder(image_encoder_pretrained)
        if self.image_encoder.out_channels!=input_dim:
            raise ValueError("input_dim does not match RegNet visual features")
        self.image_projection=nn.Linear(input_dim,hidden_dim)
        self.word_embedding=nn.Embedding(len(word_to_idx),wordvec_dim)
        self.sequence_model=nn.RNN(
            wordvec_dim,hidden_dim,nonlinearity="tanh",batch_first=True
        )
        self.output_projection=nn.Linear(hidden_dim,len(word_to_idx))
        self.null_id=word_to_idx["<NULL>"]
        self.start_id=word_to_idx["<START>"]
        self.end_id=word_to_idx["<END>"]

    def visual_state(self,images):
        # Keep the pretrained encoder in inference mode throughout this
        # small demonstration; no backbone fine tuning is performed.
        self.image_encoder.eval()
        with torch.no_grad():
            features=self.image_encoder(images)
        pooled=features.mean(dim=(2,3))
        return self.image_projection(pooled).unsqueeze(0)

    def forward(self,images,captions):
        if captions.ndim!=2 or captions.shape[1]<2:
            raise ValueError("Need [batch,time] captions with at least two tokens")
        initial=self.visual_state(images)
        sequence=self.word_embedding(captions[:,:-1])
        hidden,_=self.sequence_model(sequence,initial)
        logits=self.output_projection(hidden)
        return F.cross_entropy(
            logits.reshape(-1,logits.shape[-1]),
            captions[:,1:].reshape(-1),
            ignore_index=self.null_id,
            reduction="sum"
        )/captions.shape[0]

    @torch.no_grad()
    def sample(self,images,max_length=12):
        self.eval()
        state=self.visual_state(images)
        batch=images.shape[0]
        token=torch.full((batch,1),self.start_id,dtype=torch.long,
                         device=images.device)
        generated=[]
        finished=torch.zeros(batch,dtype=torch.bool,device=images.device)
        for _ in range(max_length):
            embed=self.word_embedding(token)
            hidden,state=self.sequence_model(embed,state)
            next_id=self.output_projection(hidden[:,0]).argmax(dim=-1)
            next_id=torch.where(finished,torch.tensor(self.null_id,
                   device=images.device),next_id)
            finished |= (next_id==self.end_id)
            generated.append(next_id)
            token=next_id.unsqueeze(1)
        return torch.stack(generated,dim=1)


def decode_captions(token_ids,idx_to_word):
    sentences=[]
    for row in token_ids:
        text=[]
        for idx in row.tolist():
            word=idx_to_word.get(int(idx),"<UNK>")
            if word=="<END>":
                break
            if word not in ("<NULL>","<START>"):
                text.append(word)
        sentences.append(" ".join(text))
    return sentences


def train_captioner(model,images,captions,num_epochs=2,batch_size=2,
                    learning_rate=1e-3,device="cpu"):
    model.to(device)
    optimizer=torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],lr=learning_rate
    )
    losses=[]
    for epoch in range(num_epochs):
        model.train()
        shuffled=torch.randperm(len(images))
        total=0.
        for ids in shuffled.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss=model(images[ids].to(device),captions[ids].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite caption loss")
            loss.backward()
            optimizer.step()
            total+=float(loss.detach())*len(ids)
        mean=total/len(images)
        losses.append(mean)
        print(f"Epoch {epoch+1}: mean loss={mean:.4f}",flush=True)
    return model,losses
