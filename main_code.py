# Importing the libraries
import pandas as pd
import re
import pickle

# Importing the Dataset (reading from text file)
file = open("News_Category_Dataset.txt", "r")
news_Category_data = []
for item in file:
    item = item.split(',')
    list(item)
    news_Category_data.append(item)

# Shortening the dataset and storing it in new list(for shortening the dataset set value in range function in for loop)
c = []
for i in range(0, len(news_Category_data)):
    c.append(news_Category_data[i])
for i in range(0, len(c)):
    k = len(c[i])
    for j in range(0, k):
        c[i].append(str(c[i][j]).split(':'))

# Preparing the data (for large dataset execution may be slow)
# execute this loop until count_str not become 0 (removing str objects from the list)
for i in range(0, len(c)):
    k = len(c[i])
    x = str(type(c[i][0]))
    for j  in range(0, len(c[i])):
        if j<len(c[i]):
            l = str(type(c[i][j]))
            if l == "<class 'str'>":
                del c[i][j]
                
# Checking whether the value of count_str is 0 after excuting this loop
count_str = 0
for i in range(0, len(c)):
    for j in range(0, len(c[i])):
        sa = str(type(c[i][j]))
        if sa == "<class 'str'>":
            count_str = count_str + 1
print("count_str:", count_str)
           
# Collecting the data and flagging it(merging the headline and short-description in one string)
data_sd = []
data_sd2 = []
data_head = []
data_head2 = []
category = []
flag = 0            
for i in range(0, len(c)):
    if len(c[i]) == 6:
        data_sd.append({'data': c[i][0][1], 'flag': i})
        data_head.append({'data':c[i][1][len(c[i][1])-1], 'flag': i})
        category.append({'category': c[i][5][1], 'flag': i})
    elif len(c[i]) > 6:
        data_sd.append({'data': c[i][0][1], 'flag': i})
        data_sd2.append({'data': c[i][1][len(c[i][1])-1], 'flag': i})
        data_head.append({'data': c[i][2][len(c[i][2])-1], 'flag': i})
        category.append({'category': c[i][len(c[i])-1][1], 'flag': i})
raw_data = []           
for i in range(0, len(c)):           # For large size dataset, this section may take time to execute.
    count = 0
    for x in data_sd2:
        if x['flag'] == i:
            raw_data.append({'data': data_sd[i]['data'] +' '+ data_head[i]['data'] +' ' + x['data'], 'flag': i})
            count = count + 1
            break
    if count == 0:
        raw_data.append({'data': data_sd[i]['data'] +' '+ data_head[i]['data'], 'flag': i})

# Cleaning the text in data
category_data = []
for i in range(0, len(c)):
    data = re.sub('[^a-zA-Z]','',category[i]['category'])
    data = data.lower()
    category_data.append({'data': data, 'flag': i})
# Finding each categories of news
each_category = []  
for x in category_data:
    if x['data'] not in each_category:
        each_category.append(x['data'])
training_data = []
for i in range(0, len(c)):
    data = re.sub('[^a-zA-z.,0-9]',' ', raw_data[i]['data'])
    data = data.lower()
    training_data.append({'data': data, 'flag': i})    
new_training_data = []
for i in range(0, len(c)):
    category_c = category_data[i]['data']
    data = re.sub('[\\\]','',training_data[i]['data'])
    data = data.lower()
    new_training_data.append({'data': data, 'flag': each_category.index(category_c)})  
      
# Converting trainnig_data to Pandas DataFrame and saving it in CSV file
training_data = pd.DataFrame(new_training_data, columns=['data', 'flag'])
training_data.to_csv("train_data.csv", sep=',', encoding='utf-8')
print(training_data.data.shape)

# Data preprocessing before training the data 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#GET VECTOR COUNT
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(training_data.data)
#TRANSFORM WORD VECTOR TO TF IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#SAVE WORD VECTOR
pickle.dump(count_vect.vocabulary_, open("count_vector.pkl","wb"))
#SAVE TF-IDF
pickle.dump(tfidf_transformer, open("tfidf.pkl","wb"))


# 1 - Multinomial Naive Bayes (Training the Model) (Accuracy = 34.45 %)
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)
clf = MultinomialNB().fit(X_train, y_train)
# predicting the output
predicted = clf.predict(X_test)
#SAVE MODEL
pickle.dump(clf, open("mnb_model.pkl", "wb"))


# 2 - Multi-layer perceptron classifier  (Accuracy = 50.9 %)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
clf_neural = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.20, random_state=42)
clf_neural.fit(X_train, y_train)
# predicting the output
predicted = clf_neural.predict(X_test)
# SAVE MODEL
pickle.dump(clf_neural, open("mlp_classifier.pkl", "wb"))


# 3 - Support vector machine(Support vector classifer)  (Accuracy = 91.02 %)
from sklearn import svm
from sklearn.model_selection import train_test_split
clf_svm = svm.LinearSVC()
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.20, random_state=42)
clf_svm.fit(X_train_tfidf, training_data.flag)
# SAVE MODEL
pickle.dump(clf_svm, open("svm.pkl", "wb"))
# Predicting the output
predicted = clf_svm.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted)

# Calculating the accuracy of model(calculating wrong predictions and accuracy)
wrong_predictions = 0
right_predictions = 0
count = 0
for predicted_item, result in zip(predicted, y_test):
    count = count + 1
    if each_category[predicted_item] != each_category[result]:
        wrong_predictions = wrong_predictions + 1
    else:
        right_predictions = right_predictions + 1
print("Right Predictions:", right_predictions)
print("Wrong Predictions:", wrong_predictions)
print("Accuracy:", ((count-wrong_predictions)/count)*100,"%")

# Saving the result of models to csv file
# saving result of multinomial Naive bayes
result_bayes = pd.DataFrame( {'true_category': y_test,'predicted_category': predicted})
result_bayes.to_csv('res_mnb.csv', sep = ',')

# saving result of multi-layer perceptron classfier
result_MLP = pd.DataFrame( {'true_category': y_test,'predicted_categorys': predicted})
result_MLP.to_csv('res_MLP.csv', sep = ',')

# saving result of SVM
result_svm = pd.DataFrame( {'true_category': y_test,'predicted_category': predicted})
result_svm.to_csv('res_svm.csv', sep = ',')

# For loading the model
loaded_model = pickle.load(open("mnb_model.pkl","rb"))

# For loading count_vector
loaded_vec = CountVectorizer(vocabulary = pickle.load(open("count_vector.pkl", "rb")))