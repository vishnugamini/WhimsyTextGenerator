from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import torch
from model_architecture import BigramLanguageModel
import encode_decode

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = BigramLanguageModel()
generator.load_state_dict(torch.load('gpt.pth', map_location=torch.device('cpu')))
generator.to(device)
generator.eval()
decode = encode_decode.EncDec()

def generate_text_stream(context, max_length,raw):
    yield raw.encode('utf-8')
    for text in generator.generate(context, max_new_tokens=max_length):
        yield text.encode('utf-8')  

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    initial_text = data['initial_text']
    max_length = int(data['max_length'])

    encoded_context = decode.encode(initial_text)
    length = len(encoded_context)
    t = torch.tensor(encoded_context, dtype=torch.long).to(device)
    context = t.view((1, length))
    
    return Response(generate_text_stream(context, max_length,initial_text), content_type='text/plain; charset=utf-8')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
