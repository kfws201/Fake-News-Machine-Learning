# Fake-News-Machine-Learning

The code provided builds a machine learning model which can be used to classify/detect fake news. The data used to train the model is too large to upload to GitHub, but it can be downloaded from this link: https://www.kaggle.com/datasets/rajatkumar30/fake-news?resource=download

You can use the pickled model on unseen data and it will classify the text as either real or fake. I provide the steps for using the model on unseen data below:

1) Load in the pickled model e.g. loaded_model = pickle.load(open('logisticregression_model.pkl', 'rb'))
2) Save your chosen text/article as a string
3) Clean the text so it can be used in the model i.e convert to lower case, remove punctuation and stopwords
4) vectorise the text
5) create a list of strings called labels of real and fake i.e labels = ['Real', 'Fake']
6) use .predict on the pickled model to get the predicted y values
7) use .predict_proba on the model to get the probability
8) use this code to print the classification and probability:
   print(f'Predicted class = {labels[y_predict[0]]}')
   print(f'Probability of the article being real = {probs[0,y_predict[0]]}')

An example of this is in my jupyter notebook script, using unseen data from NBC news. 


