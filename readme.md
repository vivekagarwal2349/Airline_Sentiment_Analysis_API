# Airline sentiment analysis API
It's an Airline sentiment analysis API server on python3 using Pyramid web framework.

## Preprocessing and cleaning

1) Lowered case all the text
2) Removed all hashtags, stop words, extra space, digits
3) Removed punctuations and lemmatizing text
4) Changed positive sentiment 1 and negative sentiment to 0.

## Logistic Regression

testing accuracy: 0.91

Performed hyper tuning of parameters using GridSearchCV. Tried parameters like 'penalty', learning rate, and 'solver' etc. 

## Set-up the virtual environment :

1. `pip3 install virtualenv` (if you don't have virtualenv installed)
2. `virtualenv airline_env`
3. `source airline_env/bin/activate`
4. `cd airline/`
5. `pip install -r requirements.txt`

To start the API server -
`python3 start.py`

http://localhost:6543?review={test_text}

To train the model -
`python3 airline_train.py`

To test the model -
`python3 airline_test.py`

Note - If you have trained the model again please update the model name in airline_test.py and start.py
