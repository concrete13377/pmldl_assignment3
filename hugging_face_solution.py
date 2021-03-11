from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en").to(device)

with open('eval-ru-100.txt') as f:
    test = f.readlines()

with open('answer.txt', 'w+') as f:
    for i, val in tqdm(enumerate(test), total=len(test)):
        inputs = tokenizer.encode(val, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_length=75, num_beams=16, early_stopping=True).to(device)
        seq = (tokenizer.decode(outputs[0]).replace('<pad> ', ''))

        if i != len(test) - 1:
            f.write(seq + '\n')
        else:
            f.write(seq)