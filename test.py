from gensim.models import Word2Vec as wv
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()


df = pd.read_csv("train.csv")
testexamples = pd.read_csv("test.csv")
testexamples = testexamples.head(1)
testexamples = testexamples.loc[:,["question1","question2"]]
testsamples = list(testexamples.question1)+list(testexamples.question2)
#print df.columns.values
sentence1 = df.loc[:,["question1","question2"]]
question1 = list(sentence1.question1)
#print len(question1)
question1 = question1[0:50]
question2 = list(sentence1.question2)
question2 = question2[0:50]
completelist = question1+question2
training_list=[]
for question in completelist:
    question = question.replace("?"," ")
    question = re.sub("\.+"," ",question)
#sentence2 = sentence2.replace("?","")
    filtered_words1 = [word.lower() for word in question.split() if word not in stopwords.words('english')]
#print filtered_words1,sentence1
#filtered_words2 = [word.lower() for word in sentence2.split() if word not in stopwords.words('english')]
    #stemmed_Sen1= map(lambda x: porter_stemmer.stem(x),filtered_words1)
    training_list.append(filtered_words1)
#lem_Sen1= map(lambda x: wordnet_lemmatizer.lemmatize(x),filtered_words1)
#sentences = [['first', 'sentence'], ['second', 'sentence']]
#print stemmed_Sen1
#print lem_Sen1
model = wv(training_list, min_count=1,size=200)
vocab = list(model.wv.vocab.keys())
#print type(model[vocab[0]])

#print model['invest'].shape,model['market'].shape
#print cosine_similarity(model['invest'], model['market'])
testsamples[0] = testsamples[0].replace("?", " ")
testsamples[0] = re.sub("\.+", " ", testsamples[0])
questiontest1 = [word.lower() for word in testsamples[0].split() if word not in stopwords.words('english')]
testsamples[1] = testsamples[1].replace("?", " ")
testsamples[1] = re.sub("\.+", " ", testsamples[1])
questiontest2 = [word.lower() for word in testsamples[1].split() if word not in stopwords.words('english')]
c =0
#print questiontest1
#print questiontest2
question1vec = np.zeros(200,dtype=np.float64)
for word in questiontest1:
    if word in model.wv.vocab.keys():
        question1vec = np.add(question1vec,model[word])
        c=c+1
    else:
        print word
d = 0
question2vec = np.zeros(200,dtype=np.float64)

for word in questiontest2:
    if word in model.wv.vocab.keys():
        question2vec = np.add(question2vec,model[word])
        d=d+1
    else:
        print word

#print question1vec,c
#print question2vec,d
question1vec = question1vec/c
question2vec = question2vec/d
print cosine_similarity(question1vec,question2vec)


'''

#print wv.n_similarity(filtered_words1,filtered_words2)
#sentences = map(lambda x: x.split(),sentence1)
#print sentences
# train word2vec on the two sentences
#model = wv(sentence1, min_count=1)
#for word, vocab_obj in wv.vocab.items():
 #   print word
#print model.similarity(filtered_words1,filtered_words2)
#print model.similarity(filtered_words1,filtered_words2)
#print model.n_similarity()
#print model['first']  # raw NumPy vector of a word
#cosine_similarity([1, 0, -1], [-1,-1, 0])
'''