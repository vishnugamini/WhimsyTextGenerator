import torch
import variables_loader
import model_architecture
import loss_estimator
import get_batch_data

v = variables_loader.Variables()
batch_size = v.batch_size # number of independent sequence we process in parallel for full use of gpu computation
block_size = v.block_size # our context for prediction (looks at 256 characters before prediction)
max_iters = v.max_iters
eval_interval = v.eval_interval
learning_rate = v.learning_rate
eval_iters = v.eval_iters
n_embd = v.n_embd
n_head = v.n_head
n_layer = v.n_layer
dropout = v.dropout
device = v.device

model = model_architecture.BigramLanguageModel()
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

for iters in range(max_iters):
    if iters % eval_interval == 0:
        losses = loss_estimator.estimate_loss()
        print(f"STEP = {iters}: train loss = {losses['train']}, val loss = {losses['val']}")
    xb,yb = get_batch_data.get_batch('train')
    logits,loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'gptv2.pth')
print("Model saved successfully.")
