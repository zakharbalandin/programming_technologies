import nltk
from nltk.tokenize import word_tokenize

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
sentence = 'one Get it from your two employer It!x27s better and cheaper than what you can get in the market. You are ! allowed to supplement  this may or may not make sense but your final coverage will ! be a lot cheaper if you start with employer ? insurance'

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

print(counting_punctuation_marks(sentence))
print(tagging_words_from_sentence(sentence))
