import pandas as pd
import re, time
import nltk
import pickle
# nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


from sklearn.neighbors import KNeighborsClassifier

class TrainAirlineSentiment(object):
    def __init__(self, filename):
        self.filename = filename

    def readfile(self, filename):
        df = pd.read_csv(filename)
        return df

    def data_cleaning(self):
        df = self.readfile(self.filename)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # mood_count=df['airline_sentiment'].value_counts()
        # print(mood_count)
        df['text'] = df['text'].map(lambda x:re.sub('@\w*','',str(x))) # remove @words
        df['text'] = df['text'].map(lambda x:re.sub('[^a-zA-Z]',' ',str(x))) #remove special character
        df['text'] = df['text'].map(lambda x:re.sub('http.*','',str(x))) #remove link
        df['text'] = df['text'].map(lambda x:str(x).lower()) #convert to lower
        df['text'].head()
        return self.remove_stopwords(df)

    def remove_stopwords(self, df):
        corpus = []
        empty = df['text'].map(lambda x:corpus.append(' '.join([word for word in str(x).strip().split() if not word in set(stopwords.words('english'))])))
        return corpus, df

    def feature_extractor(self, X_train, X_test):
        v = CountVectorizer(analyzer = "word")
        X_train_word_feature = v.fit_transform(X_train['comment_text']).toarray()
        X_test_word_feature = v.transform(X_test['comment_text']).toarray()
        vec_file = 'vectorizer.pickle'
        pickle.dump(v, open(vec_file, 'wb'))
        return X_train_word_feature, X_test_word_feature

    def data_preparation(self):
        corpus, df = self.data_cleaning()
        X = pd.DataFrame(data=corpus,columns=['comment_text'])
        y = df['airline_sentiment'].map({'negative':-1,'positive':1})
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
        print("training", len(X_train),len(y_train))
        print("testing", len(X_test),len(y_test))
        X_train_word_feature, X_test_word_feature = self.feature_extractor(X_train, X_test)
        return X_train_word_feature, X_test_word_feature, y_train, y_test

    def train(self):
        X_train_word_feature, X_test_word_feature, y_train ,y_test = self.data_preparation()
        model = LogisticRegression()
        model.fit(X_train_word_feature,y_train)
        name = "airline_sentiment_model_" + time.strftime("%Y%m%d%H%M%S") + ".sav"
        pickle.dump(model, open(name, 'wb'))
        # testing
        y_pred = model.predict(X_test_word_feature)
        cm = confusion_matrix(y_test,y_pred)
        acc_score = accuracy_score(y_test,y_pred)
        print(classification_report(y_test,y_pred),'\n Confusion Matrix:\n',cm,'\n\nAccuracy:',acc_score)



if __name__ == '__main__':
    file_name = "airline_sentiment_analysis.csv"
    print("Please wait model is training...")
    TrainAirlineSentiment(file_name).train()
