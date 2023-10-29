from profanity_filter import ProfanityFilter

pf = ProfanityFilter()
for i in range(len(data)):
    for j in range(5):

        word_counter = data['comments'][i][j]['text'].split(' ')

        for k in range(len(word_counter)):
            current_word = word_counter[k]

            if not(pf.is_clean(current_word)):
                data['comments'][i][j]['score'] = 0
                print(data['comments'][i][j]['score'])

def dirt_tongue(sentence):
    word_counter = sentence.split()
    for k in range(len(word_counter)):
        current_word = word_counter[k]

        if not (pf.is_clean(current_word)):
            dirt_score = 0
        else:
            dirt_score = 1
    
    return dirt_score
