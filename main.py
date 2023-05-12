import torch
import transformers
import json

model_path = "./pytorch_model.bin"
config_path = "./config.json"
vocab_path = "./vocab.txt"

# Check Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load models and tokenizers
config = transformers.BertConfig.from_json_file(config_path)
model = transformers.BertForSequenceClassification(config).to(device)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
tokenizer = transformers.BertTokenizerFast(vocab_path)

# Transform input
text = """
Excited to announce that I've hired a new CEO for X/Twitter. She will be starting in ~6 weeks!

My role will transition to being exec chair & CTO, overseeing product, software & sysops
"""
encoded_input = tokenizer(text, return_tensors='pt').to(device)

# Get Predictions
outputs = model(**encoded_input)
outputs = outputs.logits.softmax(dim=-1).tolist()[0]

output = {
    "sadness": outputs[0],
    "joy": outputs[1],
    "love": outputs[2],
    "anger": outputs[3],
    "fear": outputs[4],
    "surprise": outputs[5]
}

# transform to json
data = json.dumps(output, sort_keys=True, indent=4)
print(data)