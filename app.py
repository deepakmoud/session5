import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

import re


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# Download the WordNet corpus
nltk.download('wordnet')
nltk.download('omw-1.4')
# Create a lemmatizer object
lemmatizer = nltk.WordNetLemmatizer()
app = Flask(__name__)
model = pickle.load(open('Decision_tree_education_model.pkl','rb'))

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
print(cv)
corpus=pd.read_csv('corpus_dataset.csv')
corpus1=corpus['corpus'].tolist()
X = cv.fit_transform(corpus1).toarray()

@app.route('/')
def welcome():
  
    return render_template("index.html")


@app.route('/review', methods=['GET'])
def review():
    text = request.args.get('text')
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    print(review)

    text = [review] 
    
    input_data = cv.transform(text).toarray()
    
    
    input_pred = model.predict(X)
    input_pred = input_pred.astype(int)
    print(input_pred)

    
    if input_pred[0]==1:
        result= "Movie Review is Positive"
    else:
        result="Movie Review is negative" 

    return render_template('index.html', prediction_text='Movie Review Anlaysis: {}'.format(result))

if __name__=="__main__":
  app.run()
