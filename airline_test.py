import pandas as pd
import re, pickle
from nltk.corpus import stopwords
# nltk.download("stopwords")

class TestAirlineSentiment(object):
    def __init__(self, model_name, review):
        self.model_name = model_name
        self.review = review

    def data_cleaning(self, review):
        review = re.sub('@\w*','',str(review)) # remove @words
        review = re.sub('[^a-zA-Z]',' ',str(review)) #remove special character
        review = re.sub('http.*','',str(review)) #remove link
        review = str(review).lower().strip() #convert to lower
        return self.remove_stopwords(review)

    def remove_stopwords(self, review):
        filtered_review = []
        empty = filtered_review.append(' '.join([word for word in str(review).split() if not word in set(stopwords.words('english'))]))
        return filtered_review

    def test(self):
        final_review = self.data_cleaning(self.review)
        df = pd.DataFrame(data=final_review,columns=['comment_text'])
        loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
        word_feature =  loaded_vectorizer.transform(df['comment_text']).toarray()
        loaded_model = pickle.load(open(self.model_name, 'rb'))
        res = loaded_model.predict(word_feature)
        if res[0] == 1:
            return "Positive"
        else:
            return "Negative"


if __name__ == '__main__':
    model_name = "airline_sentiment_model.sav"
    review = "@AIRLINE $ % StAff I love the services http://www.google.com"
    print(TestAirlineSentiment(model_name, review).test())
