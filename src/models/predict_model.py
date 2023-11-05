from transformers import T5ForConditionalGeneration, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained('./models')

input = input()
tokens = tokenizer(
    input,
    padding="max_length",
    max_length=50,
    truncation=True,
    return_tensors="pt",
)

out = model.generate(
    input_ids = tokens["input_ids"],
    attention_mask = tokens["attention_mask"],
    max_length= 50,
    num_return_sequences = 1
)

output = [tokenizer.decode(
    gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True
) for gen_id in out]
print(output)

