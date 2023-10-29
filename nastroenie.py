import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

sentence = 'one Get it from your two employer It!x27s awful better and cheaper than what you can get in the bad market. You are ! allowed to supplement  this may or may not make sense but your final coverage will ! be a lot cheaper if you start with employer ? insurance'

def mood_of_the_text(sentence):

    tokens = tokenizer.encode(sentence, return_tensors='pt')
    result = model(tokens)

    correlation = int(torch.argmax(result.logits))

    return correlation

print(mood_of_the_text(sentence))
