import torch
import variables_loader
import encode_decode

v = variables_loader.Variables()
e = encode_decode.EncDec()
batch_size = v.batch_size
block_size = v.block_size
device = v.device


with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {}
itos = {}
for x in range(len(chars)):
    # text -> token
    stoi[chars[x]] = x
    # token -> text
    itos[x] = chars[x]


data = torch.tensor(e.encode(text), dtype = torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  if split == "train":
    data = train_data
  else:
    data = val_data
  
  ix = torch.randint(len(data) - block_size,(batch_size,))
  x = torch.stack([data[i:i + block_size] for i in ix])
  y = torch.stack([data[i+1:i + block_size+1] for i in ix])
  x,y = x.to(device),y.to(device)
  return x,y