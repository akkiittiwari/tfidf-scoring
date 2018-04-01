from tfidf import *

zipfilename = sys.argv[1]
summarizefile = sys.argv[2]

corpus = load_corpus(zipfilename)
f = open(summarizefile)
xmltext = f.read()
f.close()

tfidf = compute_tfidf(corpus)

final_scores = summarize(tfidf, xmltext, 20)

for pair in final_scores:
    print pair[0] + " " + '{0:.3f}'.format(pair[1])
