"""Offline public-edition vanilla RNN captioner forward and sampling checks."""
import unittest
import torch
from captioning_model import CaptioningRNN,decode_captions


class PortfolioCaptionTests(unittest.TestCase):
    def test_real_backbone_and_rnn_shapes(self):
        torch.set_num_threads(2)
        vocab={"<NULL>":0,"<START>":1,"<END>":2,"<UNK>":3,"cat":4}
        model=CaptioningRNN(vocab,input_dim=400,wordvec_dim=8,hidden_dim=8,
                            image_encoder_pretrained=False)
        model.image_encoder.requires_grad_(False)
        images=torch.rand(2,3,112,112)
        captions=torch.tensor([[1,4,2,0],[1,4,2,0]])
        loss=model(images,captions)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.output_projection.weight.grad)
        samples=model.sample(images,max_length=4)
        self.assertEqual(tuple(samples.shape),(2,4))
        texts=decode_captions(samples,{v:k for k,v in vocab.items()})
        self.assertEqual(len(texts),2)


if __name__=="__main__":
    unittest.main()
