import torch
import model_architecture
import encode_decode

decode = encode_decode.EncDec()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = model_architecture.BigramLanguageModel()

model.load_state_dict(torch.load('gpt.pth', map_location=torch.device('cpu')))

model.to(device)
model.eval()
print("Model loaded successfully.")
print(f"Model device: {next(model.parameters()).device}")

given = True
cont = input("enter context (just press enter if you want the generation to be random):  ")
if cont == "":
    context = torch.zeros((1,1),dtype=torch.long, device=device)
    given = False
    
if given:
    encoded_context = decode.encode(cont)
    length = len(encoded_context)
    t = torch.tensor(encoded_context)
    context = t.view((1,length))

print(f"Context tensor device: {context.device}")

token_count = int(input("no of tokens to be printed: "))
print("Generated text:")
print("----------------------------------------------------------------")
print(cont,end="")
generated_text = model.generate(context, max_new_tokens= token_count)
decoded_text = decode.decode(generated_text[0].tolist())
# print(decoded_text)
