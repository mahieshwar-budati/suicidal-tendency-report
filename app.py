#=================flask code starts here
from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
import os, random
from werkzeug.utils import secure_filename
from distutils.log import debug
from fileinput import filename
import pandas as pd
import smtplib 
from email.message import EmailMessage
from datetime import datetime
from werkzeug.utils import secure_filename
import sqlite3

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import nltk
from string import punctuation
import os
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn.metrics import accuracy_score
import pickle
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from genetic_selection import GeneticSelectionCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


UPLOAD_FOLDER = os.path.join('static', 'uploads')
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'welcome'

#define object to remove stop words and other text processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def POS(sentence):
    output = ""
    words = nltk.word_tokenize(sentence)
    words = [word for word in words if word not in stop_words]
    tags = pos_tag(words)
    for word, tag in tags:
        output += word+" "+tag+" "
    output = output.strip()
    return output

def getStatistics(sentence):
    sentence = sentence.strip()
    total_sentences = len(sent_tokenize(sentence))
    words = len(nltk.word_tokenize(sentence))
    paragraphs = len(sentence.split('\n\n'))
    return total_sentences, words, paragraphs, len(sentence)

def getTopics(sentence):
    temp = sent_tokenize(sentence)
    topics = ""
    try:
        tfidf = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
        temp = tfidf.fit_transform(temp).toarray()
        feature_names = tfidf.get_feature_names()
        lda = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online', learning_offset=50.,random_state=0)
        lda.fit(temp)
        no_top_words = 10
        for topic_idx, topic in enumerate(lda.components_):        
            words = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
            topics += words
            break
    except Exception:
        pass    
    return topics

dataset = pd.read_csv("Dataset/Suicide_Detection.csv", nrows=10000)

text_sentences = dataset['text'].ravel()#get sentences and classes from dataset
classes = dataset['class'].ravel()
labels = np.unique(dataset['class'])
original_X = []
linguistic_X = []
Y = []
statistics = []
if os.path.exists("model/X.npy"):
    #if data already processed then load all such as original features, linguistics features and statistics features
    original_X = np.load("model/X.npy")
    Y = np.load("model/Y.npy")
    statistics = np.load("model/statistics.npy")
    linguistic_X = np.load("model/linguistic.npy")
else: #if not process then loop each sentences and then process
    for i in range(len(text_sentences)):
        sentence = str(text_sentences[i]).strip()#read sentence
        if len(sentence) > 0:
            #get sentences statistics
            total_sentences, total_words, total_paragraphs, total_characters = getStatistics(sentence)
            topics = getTopics(sentence)#get topics words
            pos = POS(sentence)#get POS
            pos += topics
            pos = cleanText(pos.lower().strip())#clean all original text by merging POS, topics and sentence 
            sentence = sentence.lower().strip()+" "+topics.lower().strip()
            sentence = cleanText(sentence)#clean linguistic sentence by merging sentence and topics
            original_X.append(pos) #create array with original and libguistic features
            linguistic_X.append(sentence)
            statistics.append([total_sentences, total_words, total_paragraphs, total_characters])
            if classes[i].strip().lower() == 'suicide':
                Y.append(1)
            else:
                Y.append(0)
    original_X = np.asarray(original_X)
    linguistic_X = np.asarray(linguistic_X)
    Y = np.asarray(Y)
    statistics = np.asarray(statistics)
    np.save("model/X", original_X)
    np.save("model/Y", Y)
    np.save("model/statistics", statistics)
    np.save("model/linguistic", linguistic_X)    
print("Dataset Cleaning & Processing Completed")
print("Total Posts Found in Dataset = "+str(original_X.shape[0]))

names, count = np.unique(Y, return_counts=True)
height = count
bars = labels

#convert both original and linguistic features into TFIDF vector to extract features from text data by uinsg Ngram technique
original_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=196)
original_X = original_vectorizer.fit_transform(original_X).toarray()
original_X = np.hstack([original_X, statistics])#merging statistics features to original text vector
linguistic_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=160)
linguistic_X = linguistic_vectorizer.fit_transform(linguistic_X).toarray()
print("Features Exatrcted from TEXT using TFIDF & NGRAM for both Original & Linguistics = "+str(original_X))

def runOriginalGA():
    global original_X, Y
    if os.path.exists("model/original_ga.npy"):
        selector = np.load("model/original_ga.npy")
    else:
        estimator = RandomForestClassifier()
        #defining genetic alorithm object
        selector = GeneticSelectionCV(estimator, cv=5, verbose=1, scoring="accuracy", max_features=86, n_population=10, crossover_proba=0.5, mutation_proba=0.2,
                                      n_generations=5, crossover_independent_proba=0.5, mutation_independent_proba=0.05, tournament_size=3, n_gen_no_change=10,
                                      caching=True, n_jobs=-1)
        ga_selector = selector.fit(original_X, Y) #train with GA weights
        selector = ga_selector.support_
        np.save("model/original_ga", selector)
    return selector

def runLinguisticGA():
    global linguistic_X, Y
    if os.path.exists("model/linguistic_ga.npy"):
        selector = np.load("model/linguistic_ga.npy")
    else:
        estimator = RandomForestClassifier()
        #defining genetic alorithm object
        selector = GeneticSelectionCV(estimator, cv=5, verbose=1, scoring="accuracy", max_features=59, n_population=10, crossover_proba=0.5, mutation_proba=0.2,
                                      n_generations=5, crossover_independent_proba=0.5, mutation_independent_proba=0.05, tournament_size=3, n_gen_no_change=10,
                                      caching=True, n_jobs=-1)
        ga_selector = selector.fit(linguistic_X, Y) #train with GA weights
        selector = ga_selector.support_
        np.save("model/linguistic_ga", selector)
    return selector

original_ga = runOriginalGA()
linguistic_ga = runLinguisticGA()
original_X = original_X[:,original_ga]
linguistic_X = linguistic_X[:,linguistic_ga]
print("Original Text Features Size after applying GA = "+str(original_X.shape[1]))
print("Linguistic Text Features Size after applying GA = "+str(linguistic_X.shape[1]))

original_X_train, original_X_test, original_y_train, original_y_test = train_test_split(original_X, Y, test_size=0.2) #split dataset into train and test
linguistic_X_train, linguistic_X_test, linguistic_y_train, linguistic_y_test = train_test_split(linguistic_X, Y, test_size=0.2) #split dataset into train and test
print("Dataset Train & Test Split")

rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
xg = XGBClassifier(max_depth=60, eta=0.05)
#grouping multiple ensemble algorithm and then input to single voting classifier to train Linguistic GA selected features
linguistic_extension_model = VotingClassifier(estimators=[('rf', rf), ('dt', dt), ('xg', xg)], voting='hard')
linguistic_extension_model.fit(linguistic_X_train, linguistic_y_train)
predict = linguistic_extension_model.predict(linguistic_X_test)

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/notebook')
def notebook():
    return render_template('SuicidalDetection.html')



@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():
    if request.method == 'POST':
        f = request.files.get('file')
        data_filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],data_filename))
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'],data_filename)
        data_file_path = session.get('uploaded_data_file_path', None)
        testData = pd.read_csv(data_file_path, sep=";")
        testData = testData.values
        temp = []
        for i in range(len(testData)):
            sentence = testData[i,0] #get sentence
            topics = getTopics(sentence)#get topic
            sentence = sentence.lower().strip()+" "+topics.lower().strip()#merge both topic and sentences as linguistic features
            sentence = cleanText(sentence)
            temp.append(sentence)
        temp = linguistic_vectorizer.transform(temp).toarray()#generatee TFIDF with NGram to extract linguistic features
        temp = temp[:,linguistic_ga]#apply Genetic Algorithm to select relevant features
        predict = linguistic_extension_model.predict(temp)#apply linguistic extension Voting Classifier to predict sucidial ideation
        output = ""
        for i in range(len(predict)):
            output += "Test Data = "+str(testData[i])+" <font size='3' color='blue'>Predicted As =====> "+labels[predict[i]]+"</font><br/><br/>"
        return render_template('result.html', msg=output)

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "myprojectstp@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("myprojectstp@gmail.com", "paxgxdrhifmqcrzn")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict_lo', methods=['POST'])
def predict_lo():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("home.html")
    else:
        return render_template("signin.html")


    
if __name__ == '__main__':
    app.run()