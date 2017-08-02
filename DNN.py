from gensim.models import Word2Vec as wv
import numpy as np
from nltk.corpus import wordnet as wn
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import pickle
#from pyspark import SparkContext
import string
import time


import csv
#from pyspark.sql import SQLContext,Row
from multiprocessing.dummy import Pool as ThreadPool

tokeniser = WordPunctTokenizer()
punctuations = set(string.punctuation)

## for word2vec
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words


def makeFeatureVec(words):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = list(model.wv.vocab.keys())
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word.encode("utf-8") in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def clean_text(text):
    stopword = stopwords.words('english')
    words = tokeniser.tokenize(text.decode("utf8").strip().lower())
    punctuation_removal = filter(lambda x: x not in string.punctuation, words)
    result = filter(lambda x: x not in stopword, punctuation_removal)
    return result


df = pd.read_csv("train.csv")
sentence1 = df.loc[:,["question1","question2"]]
question1 = list(sentence1.question1)[40001:100000]
question2 = list(sentence1.question2)[40001:100000]
completelist = question1+question2
print len(completelist)
training_list=[]
counter = 0
'''
print wn.__class__            # <class 'nltk.corpus.util.LazyCorpusLoader'>
wn.ensure_loaded()            # first access to wn transforms it
print wn.__class__
pool = ThreadPool(8)
start_time = time.time()
results = pool.map(clean_text, completelist)
print "thread time: "+ str(time.time() - start_time)
'''
start_time = time.time()
for question in completelist:
    if counter%300==0:
        print counter, question
    counter += 1
    filtered_words1 = clean_text(question)
    training_list.append(filtered_words1)
print "time: "+ str(time.time() - start_time)
try:
    with open("output1.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(training_list)
    df = pd.DataFrame(training_list)
    df.to_csv("training_2.csv",encoding='utf-8')
    pickle.dump(training_list,"result_1.pkl")
except:
    print "error"
'''
print "Training model..."
model = wv(training_list, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
#model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
print "Saving model..."
model_name = "300features_40minwords_10context"
model.save(model_name)
'''

'''
model = wv.load("300features_40minwords_10context")
def prepare_test():
    test_set = []
    testexamples = pd.read_csv("test.csv")
    print "generating test set"
    for id,row in testexamples.iterrows():
        question1vec = makeFeatureVec(clean_text(row["question1"]))
        question2vec = makeFeatureVec(clean_text(row["question2"]))
        featureVec = np.append(question1vec,question2vec,axis=0)
        test_set.append(featureVec)
    return test_set

def prepareTraining():
    df = pd.read_csv("train.csv")
    df = df.loc[0:10000,:]
    X_train =[]
    y_train=[]
    print "generating training data"
    for id, row in df.iterrows():
        question1vec = makeFeatureVec(clean_text(row["question1"]))
        question2vec = makeFeatureVec(clean_text(row["question2"]))
        featureVec = np.append(question1vec, question2vec, axis=0)
        X_train.append(featureVec)
        y_train.append(row['is_duplicate'])
    return X_train,y_train

classifier = Sequential()
classifier.add(Dense(input_dim=600,units=200,activation='relu'))
classifier.add(Dense(units=100,activation='relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(units=200,activation='relu'))
classifier.add(Dense(units=100,activation='relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(units=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print "Training Neural Network...."
X_train, y_train = prepareTraining()
X_test = prepare_test()
classifier.fit(X_train,y_train,batch_size=32,epochs=500)

y_predict = classifier.predict(X_test)
try:
    df = pd.DataFrame(y_predict)
    df.to_csv("predict.csv")
except:
    print "error while writing data"
'''

