import torch
import variables_loader
import torch.nn as nn
from torch.nn import functional as F
import encode_decode
import time

decode = encode_decode.EncDec()
v = variables_loader.Variables()
n_embd = v.n_embd
block_size = v.block_size
dropout = v.dropout
vocab_size = v.vocab_size
n_head = v.n_head
n_layer = v.n_layer
device = v.device

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size))) # registering as a buffer as we dont want the optimizer to update the values  since they dont require gradients
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) #(B,T,C)
        q = self.query(x) #(B,T,C)
        v = self.value(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * (C**-0.5) #(B,T,C) @ (B,C,T) -> (B,T,T)
        tril_mask = self.tril[:T, :T]
        wei = wei.masked_fill(tril_mask== 0, float("-inf")) # (B,T,T)
        wei = torch.softmax(wei,dim=-1) # (B,T,T)
        #perform weighted aggregation of values
        wei = self.dropout(wei)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class FeedForward(nn.Module):
    
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd,n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self,x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class Block(nn.Module):
    
    def __init__(self,n_embd,n_head):
    
        super().__init__()
        head_size = n_embd// n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln1(x))
        return x
        
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #(vocab_size, vector_length)
        self.pos_embedding_table = nn.Embedding(block_size,n_embd)
        self.block = nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
#         self.block = nn.Sequential(
#             Block(n_embd, n_head = 4),
#             Block(n_embd, n_head = 4),
#             Block(n_embd, n_head = 4),
#             Block(n_embd, n_head = 4)
#             nn.LayerNorm(n_embd)
#         )
        self.lnf = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)   
    def forward(self,idx,targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.pos_embedding_table(torch.arange(T,device=device)) #(T,C)
        x = tok_emb + pos_emb #(B,T,C)
        x = self.block(x)
        x = self.lnf(x)
        logits = self.lm_head(x) #(B,T,vocab_size)
        
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-block_size:]
            logits,loss = self(idx_cond) #self goes through forward
            logits = logits[:,-1,:] #(B,C)
            probs = F.softmax(logits,dim=-1) #(B,C)
            idx_nxt = torch.multinomial(probs,num_samples=1) #(B,1)
            # idx_nxt = torch.argmax(probs,dim=-1,keepdim=True)
            # print(idx_nxt.tolist()[0])
            print(decode.decode(idx_nxt[0].tolist()),end="")
            yield decode.decode(idx_nxt[0].tolist())
            idx = torch.cat((idx,idx_nxt),dim=1) #(B,T+1)
        return idx
