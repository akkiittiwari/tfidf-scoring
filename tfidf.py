import sys

import nltk
from nltk.stem.porter import *
from sklearn.feature_extraction import stop_words
import xml.etree.cElementTree as ET
from collections import Counter
import string
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import zipfile
import os

PARTIALS = False

def gettext(xmltext):
    """
    Parse xmltext and return the text from <title> and <text> tags
    """
    xmltext = xmltext.encode('ascii', 'ignore') # ensure there are no weird char
    only_text = ''
    tree = ET.fromstring(xmltext)
    for child_of_root in tree:
        if(child_of_root.tag == 'title'):
            only_text = only_text + ' ' + child_of_root.text
        elif (child_of_root.tag == 'text'):
            for elem in tree.iterfind('.//text/*'):
                only_text = only_text + ' ' + elem.text
    return only_text


def tokenize(text):
    """
    Tokenize text and return a non-unique list of tokenized words
    found in the text. Normalize to lowercase, strip punctuation,
    remove stop words, drop words of length < 3.
    """
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)  # delete stuff but leave at least a space to avoid clumping together
    words = nltk.tokenize.word_tokenize(nopunct)
    words = [w for w in words if len(w) > 2]  # ignore a, an, to, at, be, ...
    words = [w.lower() for w in words]
    words = [w for w in words if w not in stop_words.ENGLISH_STOP_WORDS]
    return words



def stemwords(words):
    """
    Given a list of tokens/words, return a new list with each word
    stemmed using a PorterStemmer.
    """
    stemmer = PorterStemmer()
    words = [w.decode('ascii', 'ignore') for w in words]
    stemmed = [stemmer.stem(w) for w in words]
    return stemmed


def tokenizer(text):
    return stemwords(tokenize(text))


def compute_tfidf(corpus):
    """
    Create and return a TfidfVectorizer object after training it on
    the list of articles pulled from the corpus dictionary. The
    corpus argument is a dictionary mapping file name to xml text.
    """
    content = [corpus[ele] for ele in corpus]
    tfidf = TfidfVectorizer(input='content',
                            analyzer='word',
                            preprocessor=gettext,
                            tokenizer=tokenizer,
                            stop_words='english',
                            decode_error='ignore')
    tfidf = tfidf.fit(content)
    return tfidf


def summarize(tfidf, text, n):
    """
    Given a trained TfidfVectorizer object and some XML text, return
    up to n (word,score) pairs in a list.
    """
    final_scores = list()
    feature_names = tfidf.get_feature_names()
    tfidf_mat = tfidf.transform([text])
    tfidf_indeces = tfidf_mat.nonzero()[1]

    for i in range(len(tfidf_indeces)):
        final_scores.append((feature_names[tfidf_indeces[i]], tfidf_mat[0, tfidf_indeces[i]]))

    final_scores = sorted(final_scores, key=lambda t: t[1] * -1)[:n]
    final_scores = [f for f in final_scores if f[1] >= 0.090]
    return final_scores


def load_corpus(zipfilename):
    """
    Given a zip file containing root directory reuters-vol1-disk1-subset
    and a bunch of *.xml files, read them from the zip file into
    a dictionary of (word,xmltext) associations. Use namelist() from
    ZipFile object to get list of xml files in that zip file.
    Convert filename reuters-vol1-disk1-subset/foo.xml to foo.xml
    as the keys in the dictionary. The values in the dictionary are the
    raw XML text from the various files.
    """
    file_text_dict = dict()
    zip_obj = zipfile.ZipFile(zipfilename)
    filenames = zip_obj.namelist()
    for file in filenames:
        if file.endswith('.xml'):
            file_obj = zip_obj.open(file)
            file_content = file_obj.read()
            file_text_dict[file.split('/')[1]] = file_content

    return file_text_dict


