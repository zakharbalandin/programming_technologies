import yake

extractor = yake.KeywordExtractor(lan = "en",     # язык
                                  n = 3,          # максимальное количество слов в фразе
                                  dedupLim = 0.3, # порог похожести слов
                                  top = 10)        # количество ключевых слов

sentence = 'one Get it from your two employer It!x27s better and cheaper than what you can get in the market. You are ! allowed to supplement  this may or may not make sense but your final coverage will ! be a lot cheaper if you start with employer ? insurance'

def key_words(sentence):
    key_tuples = extractor.extract_keywords(sentence)
    key_words = []

    for tuple in key_tuples:
        key_words.append(tuple[0])

    return key_words

print(key_words(sentence))
