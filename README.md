# TFIDF scoring

The goal of this project is to learn a core technique used in text analysis called TFIDF or term frequency, inverse document frequency. I have used a bag-of-words representation where the order of words in a document doesn't matter--we care only about the words and how often they occur. A word's TFIDF value is often used as a feature for document clustering or classification. The more a term helps to distinguish its enclosing document from other documents, the higher its TFIDF score. As such, words with high TFIDF scores are often very good summarizing keywords for document.

Our "most common word" mechanism is simple and pretty effective but not as good as we can do. We need to penalize words that are not only common in that article but common across articles. E.g., said and price probably don't help to summarize an article as they are very common words.
 
I have used Reuters articles which are in XML format and have been parsed from there. Other natural language processing techinques such as tokenizing, removing stop words and stemming have also been used in the process.

