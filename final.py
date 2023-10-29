'''
соотносимость теста с постом - n*100 баллов

цензура  - умножает на ноль весь счёт

анализ эмоций - положительная эмоция добавляет 100

использование ключевых слов - 15 баллов * (количество ключевых слов/количество слов)

количество значащих частей речи - 7 * (количество ключевых слов/количество слов)

!? - 3 * (количество ключевых слов/количество слов)
'''

k_pohozhest = 10000
k_censoreship = 0
k_key_words = 15
k_adjectives = 2
k_verbs = 3
k_adverbs = 3
k_dirt_words = 0
k_mood = 100
k_exclamation_marks = 3
k_question_marks = 3
k_numerals = 2
k_nouns = 2

vector_metric = [
k_pohozhest,
k_key_words,
k_adjectives,
k_adverbs,
k_nouns,
k_verbs,
k_numerals,
k_question_marks,
k_exclamation_marks,
#k_mood
]
import numpy
import pandas as pd
import re
import difflib
import spacy
import yake
import nltk
import torch
import json

from nltk.tokenize import word_tokenize
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from better_profanity import profanity



def clearning(data):
    for comments in data['comments']:
        for comment in comments:
            comment['text'] = re.sub(r"[^a-zA-Z0-9 !?]", "", comment['text'])
    return data

def lemmatization(sentence):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])   # Инициализируем пространственную модель 'en', сохранив только компонент теггера, необходимый для лемматизации
    doc = nlp(sentence)
    sentence = " ".join([token.lemma_ for token in doc])    # Лемматизируем по слову и добавляем в предложение
    return sentence

def similarity(s1, s2):
    normalized1 = s1.lower()
    normalized2 = s2.lower()
    matcher = difflib.SequenceMatcher(None, normalized1, normalized2)
    return matcher.ratio()

def key_words(sentence):
    key_tuples = extractor.extract_keywords(sentence)
    key_words = []

    for tuple in key_tuples:
        key_words.append(tuple[0])

    return key_words

def tagging_words_from_sentence(sentence):
    adjectives_count, adverbs_count, nouns_count, verbs_count, numerals_count = 0, 0, 0, 0, 0

    words_in_sentence = word_tokenize(sentence)
    tags_of_words = nltk.pos_tag(words_in_sentence)

    for word in tags_of_words:
        if  word[1] == 'JJ':
            adjectives_count += 1
        if  word[1] == 'RB':
            adverbs_count += 1
        if  word[1] == 'NN':
            nouns_count += 1
        if  word[1] == 'VB':
            verbs_count += 1
        if  word[1] == 'CD':
            numerals_count += 1

    return {'adjectives_count': adjectives_count, 'adverbs_count': adverbs_count, 'nouns_count': nouns_count, 'verbs_count': verbs_count, 'numerals_count': numerals_count}

def counting_punctuation_marks(sentence):
    words = sentence.split()
    question_marks_count, exclamation_marks_count = 0, 0

    for word in words:
        for symb in word:
            if symb == '!':
                question_marks_count += 1
            if symb == '?':
                exclamation_marks_count += 1

    return {'question_marks_count': question_marks_count, 'exclamation_marks_count': exclamation_marks_count}

def mood_of_the_text(sentence):

    tokens = tokenizer.encode(sentence, return_tensors='pt')
    result = model(tokens)

    correlation = int(torch.argmax(result.logits))

    return 0

def dirt_tongue(sentence):
    word_counter = sentence.split()
    dirt_score = 0
    for k in range(len(word_counter)):
        current_word = word_counter[k]

        if (profanity.censor(current_word)[0]=='*'):
            dirt_score = 1
        else:
            dirt_score = 0

    return dirt_score
def scores_result(list):
    list_copy = list.copy()
    list.sort()
    list.reverse()
    scores_result = []
    for i in range(len(list)):
        for j in range(len(list)):
            if list_copy[i] == list[j]:
                scores_result.append(j)
    return scores_result

data = pd.read_json('ranking_train.jsonl', lines=True)
pd.options.mode.chained_assignment = None

data = data.iloc[:10000]
#data = clearning(data)

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

extractor = yake.KeywordExtractor(lan = "en",     # язык
                                  n = 1,          # максимальное количество слов в фразе
                                  dedupLim = 0.3, # порог похожести слов
                                  top = 10)        # количество ключевых слов

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

all_count_of_key_words, all_count_of_adjectives, all_count_of_adverbs, all_count_of_nouns = [], [], [], []
all_count_of_verbs, all_count_of_numerals, all_count_of_question_marks, all_count_of_exclamation_marks = [], [], [], []
all_mood_of_text, all_pohozhest, all_dirt_words = [], [], []

answer = []
for t in range(88000*5):
    answer.append([])
for i in tqdm(range(0, len(data))):
    current_total_score = [[]*len(data)]
    post = data.iloc[i]
    key_wrds = key_words(post['text'])      # ключевые слова
    texts_to_compare = []

    for dct in post['comments']:
        texts_to_compare.append(dct['text'])        # комменты без скора

    #for k in range(len(texts_to_compare)):
        #texts_to_compare[k] = lemmatization(texts_to_compare[k])    # перезаписываем коммент на лемматизированный коммент

    scores = []
    count_of_key_words = []
    count_of_adjectives = []
    count_of_adverbs = []
    count_of_nouns = []
    count_of_verbs = []
    count_of_numerals = []
    count_of_question_marks = []
    count_of_exclamation_marks = []
    mood_of_text = []
    dirt_words = []

    for text in texts_to_compare:
        scores.append(similarity(post['text'], text))   # похожесть коммента на текст поста

        count_key_words = 0
        dirt_word = []

        for word in text:       # проверка на ключевые слова
            for key_word in key_wrds:
                if similarity(word, key_word) > 0.3:        # параметр схожести (поиграться)
                    count_key_words = 1
        count_of_key_words.append(count_key_words)

        parts_of_speech = tagging_words_from_sentence(text)         # количество различных частей речи в тексте
        count_of_adjectives.append(parts_of_speech['adjectives_count'])
        count_of_adverbs.append(parts_of_speech['adverbs_count'])
        count_of_nouns.append(parts_of_speech['nouns_count'])
        count_of_verbs.append(parts_of_speech['verbs_count'])
        count_of_numerals.append(parts_of_speech['numerals_count'])

        punctuation_marks = counting_punctuation_marks(text)
        count_of_question_marks.append(punctuation_marks['question_marks_count'])
        count_of_exclamation_marks.append(punctuation_marks['exclamation_marks_count'])

        #mood_of_text.append(mood_of_the_text(text))
        dirt_words.append(not(dirt_tongue(text)))

        vector_result = [
            scores[-1],
            count_of_key_words[-1],
            count_of_adjectives[-1],
            count_of_adverbs[-1],
            count_of_nouns[-1],
            count_of_verbs[-1],
            count_of_numerals[-1],
            count_of_question_marks[-1],
            count_of_exclamation_marks[-1]
            #mood_of_text[-1]
        ]
        points = numpy.dot(vector_metric, vector_result)
        answer[i].append(points)

    all_pohozhest.append(scores)

    all_count_of_key_words.append(count_of_key_words)

    all_count_of_adjectives.append(count_of_adjectives)
    all_count_of_adverbs.append(count_of_adverbs)
    all_count_of_nouns.append(count_of_nouns)
    all_count_of_verbs.append(count_of_verbs)
    all_count_of_numerals.append(count_of_numerals)

    all_count_of_question_marks.append(count_of_question_marks)
    all_count_of_exclamation_marks.append(count_of_exclamation_marks)

    all_mood_of_text.append(mood_of_text)

    all_dirt_words.append(dirt_words)
    '''
    for l in range(len(current_total_score)):
        ball = (all_pohozhest[i][l]*k_pohozhest + all_count_of_key_words[i][l]*k_key_words + all_count_of_adjectives[i][l]*k_adjectives + all_count_of_adverbs[i][l]*k_adverbs + all_count_of_nouns[i][l]*k_nouns + all_count_of_verbs[i][l]*k_verbs + all_count_of_numerals[i][l]*k_numerals + all_count_of_question_marks[i][l]*k_question_marks + all_count_of_exclamation_marks[i][l]*k_question_marks)
        current_total_score[i][l] = ball
    '''

data['pohozhest'] = all_pohozhest
data['key_words'] = all_count_of_key_words
data['adjectives'] = all_count_of_adjectives
data['adverbs'] = all_count_of_adverbs
data['nouns'] = all_count_of_nouns
data['verbs'] = all_count_of_verbs
data['numerals'] = all_count_of_numerals
data['question_marks'] = all_count_of_question_marks
data['exclamation_marks'] = all_count_of_exclamation_marks
data['mood'] = all_mood_of_text
data['dirt_words'] = all_dirt_words
#print(answer[:3])
#print(max(answer[0]))
#print(scores_result(answer[0]))
itog_scores = []
for i in range(len(data)):
    itog_scores.append(scores_result(answer[i]))

df = pd.read_json('ranking_test.jsonl', lines=True)
df = df.iloc[:10000]

for i in range(len(df)):
    for j in range(5):
        df['comments'][i][j]['score'] = itog_scores[i][j]

df = df.to_json()
with open('D:\dataset.json', 'w') as f:
    json.dump(df, f)
