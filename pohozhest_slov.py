import pandas as pd
import re
import difflib
from tqdm import tqdm

def similarity(s1, s2):
    normalized1 = s1.lower()
    normalized2 = s2.lower()
    matcher = difflib.SequenceMatcher(None, normalized1, normalized2)
    return matcher.ratio()

pd.options.mode.chained_assignment = None
data = pd.read_json('data_frame.json', lines=True)
for comments in data['comments']:
    for comment in comments:
        comment['text'] = re.sub(r"[^a-zA-Z0-9 ]", "", comment['text'])

for i in tqdm(range(0, len(data))):
    post = data.iloc[i]
    texts_to_compare = []
    for dct in post['comments']:
        texts_to_compare.append(dct['text'])

    scores = []
    for text in texts_to_compare:
        scores.append(similarity(post['text'], text))

    for i in range(5):
        post['comments'][i]['score'] = scores[i]

    scores = sorted(scores, reverse=True)

    for i in range(5):
        post['comments'][i]['score'] = scores.index(post['comments'][i]['score'])

print(data)
