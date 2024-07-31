import torch

class Variables:
    def __init__(self):
        self.batch_size = 64 # number of independent sequence we process in parallel for full use of gpu computation
        self.block_size = 256 # our context for prediction (looks at 256 characters before prediction)
        self.max_iters = 5000
        self.eval_interval = 500
        self.learning_rate = 3e-4
        self.eval_iters = 200
        self.n_embd = 384
        self.n_head = 6
        self.n_layer = 6
        self.dropout = 0.2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vocab_size = 65
