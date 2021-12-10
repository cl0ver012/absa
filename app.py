from flask import Flask,redirect,render_template,url_for,request,jsonify
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer
from collections import Counter 
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = Flask(__name__, template_folder="template")
with open("./Models/best_lrc.pickle", 'rb') as data:
    model = pickle.load(data)
with open("./Data Engineering/PickleFiles/tfidf1.pickle", 'rb') as data:
    tfidf1 = pickle.load(data)
with open("./Data Engineering/PickleFiles/tfidf2.pickle", 'rb') as data:
    tfidf2 = pickle.load(data)

@app.route("/",methods=['GET'])
def home():
	return render_template("index.html")


@app.route('/absapredict', methods=['GET', 'POST'])
def absapredict():
    title = 'ABSA - Sentiment Analysis'
    if request.method == 'POST':
        if 'file' not in request.form or 'file_aspect' not in request.form:
            return redirect(request.url)
        file = request.form.get('file')
        file_aspect = request.form.get('file_aspect')
        if not file or not file_aspect:
            return render_template('predictor.html', title=title)
        try:
            file = request.form["file"]
            file_aspect = request.form["file_aspect"]
            
            '''stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]
            p_stemmer = PorterStemmer()
            lemmatizer = WordNetLemmatizer ()
            def decontracted(phrase):
                # specific
                phrase = re.sub(r"won't", "will not", phrase)      # replace won't with "will not"
                phrase = re.sub(r"can\'t", "can not", phrase)      # replace can or cant with 'can not'
                phrase = re.sub(r"n\'t", " not", phrase)           # replece n with 'not'
                phrase = re.sub(r"\'re", " are", phrase)           # replace re with 'are'
                phrase = re.sub(r"\'s", " is", phrase)             # replace s with 'is'
                phrase = re.sub(r"\'d", " would", phrase)          # replace 'd' with 'would'
                phrase = re.sub(r"\'ll", " will", phrase)          # replace 'll with 'will'
                phrase = re.sub(r"\'t", " not", phrase)            # replace 't' with 'not'
                phrase = re.sub(r"\'ve", " have", phrase)          # replace ve with 'have'
                phrase = re.sub(r"\'m", " am", phrase)             # replace 'm with 'am'
                return phrase

  
            def preprocess_text(text_data):
                preprocessed_text = []             
                # tqdm is for printing the status bar
                for sentance in tqdm(text_data):
                    sent = decontracted(sentance)           #calling funcion for each sentence
                    #print("1st sent" , sent)
                    sent = sent.replace('\\r', ' ')         # replace line terminator with space
                    sent = sent.replace('\\n', ' ')         # replace new line charactor with space
                    sent = sent.replace('\\"', ' ')         
                    sent = re.sub('[^A-Za-z]+', ' ', sent)  # remove anything that is not letter
                    sent = ''.join(p_stemmer.stem(token) for token in sent )
                    sent = ''.join(lemmatizer.lemmatize(token) for token in sent )
                    sent  = ' '.join(e for e in sent.split() if len( Counter(e)) > 2 )
                    #sent = lstr(emmatize_text(sent)
                    
                    sent = ' '.join(e for e in sent.split() if e.lower() not in 'root/nltk_data/corpora/stop_words') # checking for stop words
                    preprocessed_text.append(sent.lower().strip())
                return preprocessed_text
            file = preprocess_text([file])'''
            features_test_tfidf1 = tfidf1.transform([file]).toarray()
            features_test_tfidf2 = tfidf2.transform([file_aspect]).toarray()
            X = np.concatenate((features_test_tfidf1, features_test_tfidf2), axis=1)
            pred = model.predict(X)
            pred = pred[0]
            if pred== 0:
                predict = "Negative"
                return render_template('Negative_Prediction.html', prediction=predict,title=title)
            elif pred==1:
                predict = "Neutral"
                return render_template('Neutral_Prediction.html', prediction=predict,title=title)
            elif pred==2:
                predict = "Positive"
                return render_template('Positive_Prediction.html', prediction=predict,title=title)
        except:
            pass
    return render_template('predictor.html', title=title)
    

if __name__ == '__main__':
    app.run(debug=False)