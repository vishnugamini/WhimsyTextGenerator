import torch
import get_batch_data as g
import model_architecture

model = model_architecture.BigramLanguageModel()

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(200)
        for k in range(200):
            X,Y = g.get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()

        out[split] = losses.mean()
    model.train()
    return out

