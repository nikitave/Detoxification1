import pandas as pd
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from tqdm import trange
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification

data = pd.read_csv("./data/test_dataset.tsv", delimiter="\t")
data = data.drop("Unnamed: 0", axis=1)

model_name="t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained('./models')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)
output = []

for i in trange(0, len(data.reference), 128):
    tokens = tokenizer(
        list(data.reference[i:i + 128]),
        padding="max_length",
        max_length=50,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    out = model.generate(
        input_ids = tokens["input_ids"],
        attention_mask = tokens["attention_mask"],
        max_length= 50,
        num_return_sequences = 1
    )
    output += [tokenizer.decode(
        gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True
    ) for gen_id in out]


model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings1 = []
embeddings2 = []

embeddings1 = model.encode(list(data.reference), show_progress_bar=True, convert_to_tensor=True)
embeddings2 = model.encode(output, show_progress_bar=True, convert_to_tensor=True)

cosine_scores = []
for i in trange(len(embeddings1)):
    cosine_scores += util.cos_sim(embeddings1[i], embeddings2[i])

average_similiraty_score = sum(cosine_scores) / len(embeddings1)
print(float(average_similiraty_score.detach()))

model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    
def text2toxicity(text, aggregate=True):
    """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba
    
sum_toxicity = 0
for i in trange (len(data)):
    sum_toxicity += text2toxicity(data.reference[i], True)
print(sum_toxicity)

result_toxicity_average_before = sum_toxicity / len(data)
print(result_toxicity_average_before)

sum_toxicity = 0
for i in trange (len(data)):
    sum_toxicity += text2toxicity(output[i], True)
print(sum_toxicity)

result_toxicity_average_before = sum_toxicity / len(output)
print(result_toxicity_average_before)
