from tfidf import *
import sys

filename = sys.argv[1]
f = open(filename)
xmltext = f.read()
f.close()
words = tokenizer(gettext(xmltext))
ctr = Counter(words)
for pair in ctr.most_common(10):
    print pair[0] + " " + str(pair[1])
