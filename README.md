# Natural-Language-Processing
Model can be summarized in four steps:
1) HOT VECTOR GENERATION: In the given model, we will first extract sentences and the Annotator One’s response. Annotator response is then fed through the One hot calculator function which encodes the one hot value of each response.
2) POS TAGGING: The second step is to do the POS tagging of each sentence using the nltk module in Python. Result is POS tagged sentence. The generated POS tagged sentence has Key Value pair of word and Tags in it. For multiple Regression to run, we just need the Tags, so Tags are extracted from each pair and list is created.
3) APPLYING MULTIPLE REGRESSION: To apply Supervised Learning model must be trained. To Train the model, the data set is divided into sets (80:20) - Training and Test. Multiple Regression function takes two arguments, first argument is the list of Tags of each sentence and the second argument is their respective one-hot vector value. 
4) DETERMING ACCURACY: After model is trained model can predict the values for test data set. Finally, the predicted values are matched against the actual values in Test data set to determine the accuracy of the model.
