class EncDec():
    def __init__(self):
        with open("input.txt", 'r', encoding='utf-8') as f:
            self.text = f.read()
            
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        self.stoi = {}
        self.itos = {}
        for x in range(len(self.chars)):
            # text -> token
            self.stoi[self.chars[x]] = x
            # token -> text
            self.itos[x] = self.chars[x]


    def encode(self,s):
        arr = []
        for i in s:
            arr.append(self.stoi[i])
        return arr
    def decode(self,s):
        arr = []
        for i in s:
            arr.append(self.itos[i])
        return "".join(arr)