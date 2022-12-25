# Airline sentiment analysis API
It's an Airline sentiment analysis API server on python3 using Pyramid web framework.

Set-up the virtual environment :

1. run pip3 install virtualenv (if you don't have virtualenv installed)
2. run virtualenv airline_env
3. run source airline_env/bin/activate
4. run cd airline/
5. run pip install -r requirements.txt

To start the API server -
`python3 start.py`

To train the model -
`python3 airline_train.py`

To test the model -
`python3 airline_test.py`

Note - If you have trained the model again please update the model name in airline_test.py and start.py
