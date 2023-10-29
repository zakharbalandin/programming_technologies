import spacy

sentence = 'one Get it from your two employer It!x27s better and cheaper than what you can get in the market. You are ! allowed to supplement  this may or may not make sense but your final coverage will ! be a lot cheaper if you start with employer ? insurance'

def lemmatization(sentence):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) # Инициализируем пространственную модель 'en', сохранив только компонент теггера, необходимый для лемматизации
    doc = nlp(sentence)
    sentence = " ".join([token.lemma_ for token in doc]) # Лемматизируем по слову и добавляем в предложение
    return sentence

print(lemmatization(sentence))
