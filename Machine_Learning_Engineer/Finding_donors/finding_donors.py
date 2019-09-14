#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Supervised Learning
# ## Project: Finding Donors for *CharityML*

# Welcome to the second project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Please specify WHICH VERSION OF PYTHON you are using when submitting this notebook. Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# 
# In this project, you will employ several supervised algorithms of your choice to accurately model individuals' income using data collected from the 1994 U.S. Census. You will then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. Your goal with this implementation is to construct a model that accurately predicts whether an individual makes more than $50,000. This sort of task can arise in a non-profit setting, where organizations survive on donations.  Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with.  While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publically available features. 
# 
# The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). The datset was donated by Ron Kohavi and Barry Becker, after being published in the article _"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"_. You can find the article by Ron Kohavi [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf). The data we investigate here consists of small changes to the original dataset, such as removing the `'fnlwgt'` feature and records with missing or ill-formatted entries.

# ----
# ## Exploring the Data
# Run the code cell below to load necessary Python libraries and load the census data. Note that the last column from this dataset, `'income'`, will be our target label (whether an individual makes more than, or at most, $50,000 annually). All other columns are features about each individual in the census database.

# In[1]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Ignoring the warnings along the document
import warnings
warnings.filterwarnings('ignore')

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=5))


# ### Implementation: Data Exploration
# A cursory investigation of the dataset will determine how many individuals fit into either group, and will tell us about the percentage of these individuals making more than \$50,000. In the code cell below, you will need to compute the following:
# - The total number of records, `'n_records'`
# - The number of individuals making more than \$50,000 annually, `'n_greater_50k'`.
# - The number of individuals making at most \$50,000 annually, `'n_at_most_50k'`.
# - The percentage of individuals making more than \$50,000 annually, `'greater_percent'`.
# 
# ** HINT: ** You may need to look at the table above to understand how the `'income'` entries are formatted. 

# In[2]:


# Total number of records
n_records = data.shape[0]

# Number of records where individual's income is more than $50,000
n_greater_50k = data[data['income'] != '<=50K'].shape[0]

# Number of records where individual's income is at most $50,000
n_at_most_50k = n_records - n_greater_50k

# Percentage of individuals whose income is more than $50,000
greater_percent = round((n_greater_50k / n_records) * 100, 0)

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))


# ** Feature set Exploration **
# 
# * **age**: continuous. 
# * **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
# * **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
# * **education-num**: continuous. 
# * **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
# * **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
# * **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
# * **race**: Black, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other. 
# * **sex**: Female, Male. 
# * **capital-gain**: continuous. 
# * **capital-loss**: continuous. 
# * **hours-per-week**: continuous. 
# * **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

# ----
# ## Preparing the Data
# Before data can be used as input for machine learning algorithms, it often must be cleaned, formatted, and restructured — this is typically known as **preprocessing**. Fortunately, for this dataset, there are no invalid or missing entries we must deal with, however, there are some qualities about certain features that must be adjusted. This preprocessing can help tremendously with the outcome and predictive power of nearly all learning algorithms.

# ### Transforming Skewed Continuous Features
# A dataset may sometimes contain at least one feature whose values tend to lie near a single number, but will also have a non-trivial number of vastly larger or smaller values than that single number.  Algorithms can be sensitive to such distributions of values and can underperform if the range is not properly normalized. With the census dataset two features fit this description: '`capital-gain'` and `'capital-loss'`. 
# 
# Run the code cell below to plot a histogram of these two features. Note the range of the values present and how they are distributed.

# In[3]:


# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)


# For highly-skewed feature distributions such as `'capital-gain'` and `'capital-loss'`, it is common practice to apply a <a href="https://en.wikipedia.org/wiki/Data_transformation_(statistics)">logarithmic transformation</a> on the data so that the very large and very small values do not negatively affect the performance of a learning algorithm. Using a logarithmic transformation significantly reduces the range of values caused by outliers. Care must be taken when applying this transformation however: The logarithm of `0` is undefined, so we must translate the values by a small amount above `0` to apply the the logarithm successfully.
# 
# Run the code cell below to perform a transformation on the data and visualize the results. Again, note the range of values and how they are distributed. 

# In[4]:


# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed = True)


# ### Normalizing Numerical Features
# In addition to performing transformations on features that are highly skewed, it is often good practice to perform some type of scaling on numerical features. Applying a scaling to the data does not change the shape of each feature's distribution (such as `'capital-gain'` or `'capital-loss'` above); however, normalization ensures that each feature is treated equally when applying supervised learners. Note that once scaling is applied, observing the data in its raw form will no longer have the same original meaning, as exampled below.
# 
# Run the code cell below to normalize each numerical feature. We will use [`sklearn.preprocessing.MinMaxScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) for this.

# In[5]:


# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))


# ### Implementation: Data Preprocessing
# 
# From the table in **Exploring the Data** above, we can see there are several features for each record that are non-numeric. Typically, learning algorithms expect input to be numeric, which requires that non-numeric features (called *categorical variables*) be converted. One popular way to convert categorical variables is by using the **one-hot encoding** scheme. One-hot encoding creates a _"dummy"_ variable for each possible category of each non-numeric feature. For example, assume `someFeature` has three possible entries: `A`, `B`, or `C`. We then encode this feature into `someFeature_A`, `someFeature_B` and `someFeature_C`.
# 
# |   | someFeature |                    | someFeature_A | someFeature_B | someFeature_C |
# | :-: | :-: |                            | :-: | :-: | :-: |
# | 0 |  B  |  | 0 | 1 | 0 |
# | 1 |  C  | ----> one-hot encode ----> | 0 | 0 | 1 |
# | 2 |  A  |  | 1 | 0 | 0 |
# 
# Additionally, as with the non-numeric features, we need to convert the non-numeric target label, `'income'` to numerical values for the learning algorithm to work. Since there are only two possible categories for this label ("<=50K" and ">50K"), we can avoid using one-hot encoding and simply encode these two categories as `0` and `1`, respectively. In code cell below, you will need to implement the following:
#  - Use [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) to perform one-hot encoding on the `'features_log_minmax_transform'` data.
#  - Convert the target label `'income_raw'` to numerical entries.
#    - Set records with "<=50K" to `0` and records with ">50K" to `1`.

# In[6]:


# One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
features_final = pd.get_dummies(features_log_minmax_transform)

# Encode the 'income_raw' data to numerical values
income = income_raw.replace({'<=50K':0, '>50K':1})

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
# print(encoded)


# ### Shuffle and Split Data
# Now all _categorical variables_ have been converted into numerical features, and all numerical features have been normalized. As always, we will now split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing.
# 
# Run the code cell below to perform this split.

# In[7]:


# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

warnings.filterwarnings('ignore')

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# ----
# ## Evaluating Model Performance
# In this section, we will investigate four different algorithms, and determine which is best at modeling the data. Three of these algorithms will be supervised learners of your choice, and the fourth algorithm is known as a *naive predictor*.

# ### Metrics and the Naive Predictor
# *CharityML*, equipped with their research, knows individuals that make more than \$50,000 are most likely to donate to their charity. Because of this, *CharityML* is particularly interested in predicting who makes more than \$50,000 accurately. It would seem that using **accuracy** as a metric for evaluating a particular model's performace would be appropriate. Additionally, identifying someone that *does not* make more than \$50,000 as someone who does would be detrimental to *CharityML*, since they are looking to find individuals willing to donate. Therefore, a model's ability to precisely predict those that make more than \$50,000 is *more important* than the model's ability to **recall** those individuals. We can use **F-beta score** as a metric that considers both precision and recall:
# 
# $$ F_{\beta} = (1 + \beta^2) \cdot \frac{precision \cdot recall}{\left( \beta^2 \cdot precision \right) + recall} $$
# 
# In particular, when $\beta = 0.5$, more emphasis is placed on precision. This is called the **F$_{0.5}$ score** (or F-score for simplicity).
# 
# Looking at the distribution of classes (those who make at most \$50,000, and those who make more), it's clear most individuals do not make more than \$50,000. This can greatly affect **accuracy**, since we could simply say *"this person does not make more than \$50,000"* and generally be right, without ever looking at the data! Making such a statement would be called **naive**, since we have not considered any information to substantiate the claim. It is always important to consider the *naive prediction* for your data, to help establish a benchmark for whether a model is performing well. That been said, using that prediction would be pointless: If we predicted all people made less than \$50,000, *CharityML* would identify no one as donors. 
# 
# 
# #### Note: Recap of accuracy, precision, recall
# 
# ** Accuracy ** measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions (the number of test data points).
# 
# ** Precision ** tells us what proportion of messages we classified as spam, actually were spam.
# It is a ratio of true positives(words classified as spam, and which are actually spam) to all positives(all words classified as spam, irrespective of whether that was the correct classificatio), in other words it is the ratio of
# 
# `[True Positives/(True Positives + False Positives)]`
# 
# ** Recall(sensitivity)** tells us what proportion of messages that actually were spam were classified by us as spam.
# It is a ratio of true positives(words classified as spam, and which are actually spam) to all the words that were actually spam, in other words it is the ratio of
# 
# `[True Positives/(True Positives + False Negatives)]`
# 
# For classification problems that are skewed in their classification distributions like in our case, for example if we had a 100 text messages and only 2 were spam and the rest 98 weren't, accuracy by itself is not a very good metric. We could classify 90 messages as not spam(including the 2 that were spam but we classify them as not spam, hence they would be false negatives) and 10 as spam(all 10 false positives) and still get a reasonably good accuracy score. For such cases, precision and recall come in very handy. These two metrics can be combined to get the F1 score, which is weighted average(harmonic mean) of the precision and recall scores. This score can range from 0 to 1, with 1 being the best possible F1 score(we take the harmonic mean as we are dealing with ratios).

# ### Question 1 - Naive Predictor Performace
# * If we chose a model that always predicted an individual made more than $50,000, what would  that model's accuracy and F-score be on this dataset? You must use the code cell below and assign your results to `'accuracy'` and `'fscore'` to be used later.
# 
# ** Please note ** that the the purpose of generating a naive predictor is simply to show what a base model without any intelligence would look like. In the real world, ideally your base model would be either the results of a previous model or could be based on a research paper upon which you are looking to improve. When there is no benchmark model set, getting a result better than random choice is a place you could start from.
# 
# ** HINT: ** 
# 
# * When we have a model that always predicts '1' (i.e. the individual makes more than 50k) then our model will have no True Negatives(TN) or False Negatives(FN) as we are not making any negative('0' value) predictions. Therefore our Accuracy in this case becomes the same as our Precision(True Positives/(True Positives + False Positives)) as every prediction that we have made with value '1' that should have '0' becomes a False Positive; therefore our denominator in this case is the total number of records we have in total. 
# * Our Recall score(True Positives/(True Positives + False Negatives)) in this setting becomes 1 as we have no False Negatives.

# In[8]:


'''
TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case
'''
# Calculate accuracy, precision and recall
accuracy = round(np.sum(income) / income.shape[0], 4)
recall = 1
precision = accuracy

# Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
beta = 0.5
fscore = round((1+ beta**2) * (precision * recall) / ((beta ** 2 * precision) + recall), 4)

# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


# ###  Supervised Learning Models
# **The following are some of the supervised learning models that are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
# - Gaussian Naive Bayes (GaussianNB)
# - Decision Trees
# - Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - K-Nearest Neighbors (KNeighbors)
# - Stochastic Gradient Descent Classifier (SGDC)
# - Support Vector Machines (SVM)
# - Logistic Regression

# ### Question 2 - Model Application
# List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen
# 
# - Describe one real-world application in industry where the model can be applied. 
# - What are the strengths of the model; when does it perform well?
# - What are the weaknesses of the model; when does it perform poorly?
# - What makes this model a good candidate for the problem, given what you know about the data?
# 
# ** HINT: **
# 
# Structure your answer in the same format as above^, with 4 parts for each of the three models you pick. Please include references with your answer.

# **Answer: **
# 
# **1) Decision Tree Classifier:**
# 
# *Applicability in the real world:* 
# 
# - the decision tree algorithm is used to estimate if a person is fit or to decide the product to recommend to a specific customer.
# 
# *Model strenghts:* 
# 
# - easy to implement and interpret; 
# 
# - usable as Regressor and Classifier;
# 
# - inputs do not require extensive tranformation;
# 
# - performs well with large dataset;
# 
# - results are easily associated to human decisions.
# 
# *Model weaknesses:*
# 
# - is nondeterministic polynomial completed: may be influenced by local features;
# 
# - is biased towards the data with the highest amount of levels;
# 
# - trees are not robust to changes in the training dataset;
# 
# - trees generate complex results that do not generalise well (overfitting).
# 
# *Applicability to the study case:*
# 
# - the available dataset is sufficiently large (45222 rows);
# 
# - the required output is a classification (the income is greater/lower than 50K?);
# 
# - all of the dataset features have a comparable amount of levels;
# 
# - is easy to implement and the results are easy to communicate.
# 
# **2) Support Vector Machine:**
# 
# *Applicability in the real world:* 
# 
# - the support vector machine algorithm is used for face recognition, text categorisation and bioinformatics.
# 
# *Model strenghts:* 
# 
# - limits the overfitting: changes in the data that are distant from the vector do not influence the model;
# 
# - has abundance of implementations;
# 
# - useful for both linearly (hard-margin) and non-linearly (soft-margin) separated datasets;
# 
# - can be used as semi-supervised learning method: works on data that are partially labelled. 
# 
# *Model weaknesses:*
# 
# - does not consider the structure of the text in Natural Language Processing;
# 
# - works only with a two classes task and cannot be used with multiple classes task;
# 
# - does not work in large dataset, as the training time is cubic to the dataset size.
# 
# *Applicability to the study case:*
# 
# - mostly usable for classification cases with two classes;
# 
# - data are large but do not reach the treshold of 100000, suggested for the specific algorithm;
# 
# - works well with datasets that include a large amount of features (~50).
# 
# **3) K nearest neighbor:**
# 
# *Applicability in the real world:* 
# 
# - the KNN algorithm is used to provide recommendations: movies suggestion in the Netflix or Amazon Prime platforms.
# 
# *Model strenghts:* 
# 
# - is intuitive and simple;
# 
# - does not require assumptions;
# 
# - can be used as Regressor and Classifier;
# 
# - relies on a single hyper parameter that determines the distance.
# 
# *Model weaknesses:*
# 
# - is very slow, expecially with large dataset;
# 
# - requires homogeinity in the inputs;
# 
# - choosing the neighbors amount is challenging;
# 
# - is sensible to outliers.
# 
# *Applicability to the study case:*
# 
# - works well with two classes classification;
# 
# - the dataset is large enough to not cause variance errors but not enough to slow the algorithm speed;
# 
# - no assumptions need to be done on the correlation between income and predictors;
# 
# - the algorithm results are easy to present to the Charity decision makers.
# 
# *Reference materials:*
# 
# https://medium.com/machine-learning-101/chapter-3-decision-trees-theory-e7398adac567
# 
# https://www.quora.com/In-what-real-world-applications-is-the-decision-tree-classifier-used
# 
# https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72
# 
# https://www.digitalvidya.com/blog/understanding-support-vector-machines-and-its-applications/
# 
# https://data-flair.training/blogs/applications-of-svm/
# 
# A fuzzy K-nearest neighbor algorithm, James M. Keller et al., IEEE Transactions on Systems, Man, and Cybernetics (1985)
# 
# https://www.fromthegenesis.com/pros-and-cons-of-k-nearest-neighbors/

# ### Implementation - Creating a Training and Predicting Pipeline
# To properly evaluate the performance of each model you've chosen, it's important that you create a training and predicting pipeline that allows you to quickly and effectively train models using various sizes of training data and perform predictions on the testing data. Your implementation here will be used in the following section.
# In the code block below, you will need to implement the following:
#  - Import `fbeta_score` and `accuracy_score` from [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
#  - Fit the learner to the sampled training data and record the training time.
#  - Perform predictions on the test data `X_test`, and also on the first 300 training points `X_train[:300]`.
#    - Record the total prediction time.
#  - Calculate the accuracy score for both the training subset and testing set.
#  - Calculate the F-score for both the training subset and testing set.
#    - Make sure that you set the `beta` parameter!

# In[9]:


# Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # Calculate the training time
    results['train_time'] = end - start
        
    # Get the predictions on the test set(X_test), then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end - start
            
    # Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train[:300], beta = 0.5)
        
    # Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta = 0.5)
       
    # Print the amount of data used to train the model
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results


# ### Implementation: Initial Model Evaluation
# In the code cell, you will need to implement the following:
# - Import the three supervised learning models you've discussed in the previous section.
# - Initialize the three models and store them in `'clf_A'`, `'clf_B'`, and `'clf_C'`.
#   - Use a `'random_state'` for each model you use, if provided.
#   - **Note:** Use the default settings for each model — you will tune one specific model in a later section.
# - Calculate the number of records equal to 1%, 10%, and 100% of the training data.
#   - Store those values in `'samples_1'`, `'samples_10'`, and `'samples_100'` respectively.
# 
# **Note:** Depending on which algorithms you chose, the following implementation may take some time to run!

# In[10]:


# Import the three supervised learning models from sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Initialize the models
clf_A = DecisionTreeClassifier()
clf_B = KNeighborsClassifier()
clf_C = SVC(random_state=42)

clfs = [clf_A, clf_B, clf_C]

# Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_100 = len(y_train)
samples_10 = int(samples_100 * 0.1)
samples_1 = int(samples_100 * 0.01)

# Collect results on the learners
results = {}
for clf in clfs:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] =         train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the supervised learning models chosen
vs.evaluate(results, accuracy, fscore)

# Print the KPIs results
print('The DecisionTreeClassifier accuracy and F-score are {} and {} when using 100% of the training data!'.format(
    round(results['DecisionTreeClassifier'][2]['acc_test'], 3), round(
        results['DecisionTreeClassifier'][2]['f_test'], 3)))

print('The KNeighborsClassifier accuracy and F-score are {} and {} when using 100% of the training data!'.format(
    round(results['KNeighborsClassifier'][2]['acc_test'], 3), round(
        results['KNeighborsClassifier'][2]['f_test'], 3)))

print('The Support Vector Machine accuracy and F-score are {} and {} when using 100% of the training data!'.format(
    round(results['SVC'][2]['acc_test'], 3), round(
        results['SVC'][2]['f_test'], 3)))


# ----
# ## Improving Results
# In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F-score. 

# ### Question 3 - Choosing the Best Model
# 
# * Based on the evaluation you performed earlier, in one to two paragraphs, explain to *CharityML* which of the three models you believe to be most appropriate for the task of identifying individuals that make more than \$50,000. 
# 
# ** HINT: ** 
# Look at the graph at the bottom left from the cell above(the visualization created by `vs.evaluate(results, accuracy, fscore)`) and check the F score for the testing set when 100% of the training set is used. Which model has the highest score? Your answer should include discussion of the:
# * metrics - F score on the testing when 100% of the training data is used, 
# * prediction/training time
# * the algorithm's suitability for the data.

# **Answer: **
# 
# The choice of the best model to identify the individuals that earn more than 50K depends on several Key Performance Indicators (KPIs) and some considerations. As an initial step, it is useful to evaluate the fraction of true positive over the predicted positive values (precision) and the fraction of predicted positive values over the actual positive values (recall). The ratio among these indicators is called F-score and it is the first KPI used in this analysis. We estimate the F-score on the test dataset after trained a Decision Tree Classifier (DTC), K Neighbors Classifier (KNC), and Support Vector Machine (SVM) models on the data totality. This preliminary analysis leads to the following:
# - DTC: 0.629 
# - KNC: 0.639
# - SVM: 0.674
# 
# The SVM model shows the highest F-score, reaching 67.4% and hence resulting the most appropriate to use. The KNC model is the second one in the rank (63.9%), while the DTC is associated to the lowest F-score value (62.9%), although the difference between the last two might be not significant.
# 
# Having analysed the F-score estimates, we focus on the evaluation of the model accuracy. This KPI returns the fraction of correctly classified values, which is:
# - DTC: 0.819
# - KNC: 0.824
# - SVM: 0.837
# 
# In agreement with the F-score estimates, the SVM model is the most efficient, classifying approximately 84% of the data correctly. The KNC is the second in the rank, classifying correctly 1.5% less data. The DTC is the least efficient, with the accuracy being 0.819.
# 
# *Preliminary conclusion:* 
# 
# the SVC model seems to be the most efficient for the study case, being associated to the highest F-score and accuracy. This conclusion is additionally supported by the high F-score and accuracy values shown in the upper central and right graphs in the panels above for the DTC and KNC models, which highligth potential overfitting. 
# 
# *Additional considerations:*
# 
# Notwithstanding the results highlighted in the previous paragraph, the F-score and accuracy estimates cannot be used alone to decide the most efficient model. It is also important to evaluate the amount of computer power required by each model to reach an acceptable performance. This information is stored in the left graphs in the panels above and evidences that the SVC requires roughly 100 times more time to run on the training data than the other two models. The SVC is also one order of magnitude slower than the DTC model on the test data, although the KNC model is the slowest in this occasion. 
# 
# *Final conclusion:*
# 
# The SVC model is the most efficient model for the study case analysed in this jupyter notebook, but this model requires a significant amount of computer time to run. If a fast solution is required, I suggest the use of the KNC model, which reduces slighltly the accuracy and F-score but limits the requested computer time by a facotr of 1.5.   
# 
# *Algorithm suitability:*
# 
# As summarised before in this document, the SVC model is useful to solve the study case and identify the individuals that earn more the 50K. This algorithm is mostly usable for classification cases with two classes; it works quite well with large datasets that do not reach a treshold of 100000 values and have a large amount of features (around 50). 
# 
# Reference materials:
# 
# https://en.wikipedia.org/wiki/F1_score

# ### Question 4 - Describing the Model in Layman's Terms
# 
# * In one to two paragraphs, explain to *CharityML*, in layman's terms, how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical jargon, such as describing equations.
# 
# ** HINT: **
# 
# When explaining your model, if using external resources please include all citations.

# **Answer: ** 
# 
# *General introduction:*
# 
# The Support Vector Machine (SVM) is a machine learning algorithm usable to separate data classes (classification) or quantify continuous values from predictive variables (regression). Between these uses, the SVM algorithm is mostly adopted as classifier, as it is done in the present study case. 
# 
# *Algorithm summary:*
# 
# Giving a dataset that needs to be classified, the SVM algorithm identifies a hyper plane that separates the data into two classes (e.g. income higher and lower than 50K). This hyper plane can be thought as a line in a 2-D space and the associated confidence is directly correlated with the distance of the border data from the plane.
# 
# *Additional details:*
# 
# The SVM application relies on subsequent steps:
# 
# - the data are splitted randomly between a train and test datasets;
# - the train dataset is used to identify several possible hyper planes and select the one that minimises the error;
# - the hyper plane performance is tested on the test dataset to avoid model dependencies from the data on which it is built (overfitting);
# - finally, the model is applied to the entire dataset and potentially improved, as explained hereafter in this document.
# 
# Reference materials:
# 
# http://blog.aylien.com/support-vector-machines-for-dummies-a-simple/
# https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72

# ### Implementation: Model Tuning
# Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:
# - Import [`sklearn.grid_search.GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
# - Initialize the classifier you've chosen and store it in `clf`.
#  - Set a `random_state` if one is available to the same state you set before.
# - Create a dictionary of parameters you wish to tune for the chosen model.
#  - Example: `parameters = {'parameter' : [list of values]}`.
#  - **Note:** Avoid tuning the `max_features` parameter of your learner if that parameter is available!
# - Use `make_scorer` to create an `fbeta_score` scoring object (with $\beta = 0.5$).
# - Perform grid search on the classifier `clf` using the `'scorer'`, and store it in `grid_obj`.
# - Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_fit`.
# 
# **Note:** Depending on the algorithm chosen and the parameter list, the following implementation may take some time to run!

# In[11]:


# Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

# Initialize the classifier
clf = SVC(random_state=42)

# Create the parameters list you wish to tune, using a dictionary if needed.
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = {'C': [0.1, 0.5, 1.0], 'cache_size': [50, 100, 200], 'class_weight': 
              [None, 'balanced'], 'gamma': [1e-3, 1e-4]}

# Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta = 0.5)

# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(estimator = clf, param_grid = parameters, scoring = scorer )

# Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))


# ### Question 5 - Final Model Evaluation
# 
# * What is your optimized model's accuracy and F-score on the testing data? 
# * Are these scores better or worse than the unoptimized model? 
# * How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in **Question 1**?_  
# 
# **Note:** Fill in the table below with your results, and then provide discussion in the **Answer** box.

# #### Results:
# 
# |     Metric     | Unoptimized Model | Optimized Model |
# | :------------: | :---------------: | :-------------: | 
# | Accuracy Score |      0.8371       |     0.8365      |
# | F-score        |      0.6745       |     0.6804      |
# 

# **Answer: **
# 
# The optimisation has improved the overall performance of the model, although the results are comparable between the optimised and unoptimised iteration. For the former the accuracy and F-score are 83.71% and 67.45%, respectively. For the latter, the same KPIs are quantifiable as 83.65% and 68.04%.
# 
# The change in accuracy before and after the optimisation seems to suggest that the unoptimised model performs better than the optimised results. This may be due to the choice of the values given as parameters to the GridSearchCV algorithm, that are less efficient than the ones used as default in the model. However, a variation of 0.06% in the accuracy score may be considered not significant for our scope.
# 
# The change in F-score before and after the model optimisation identifies an improvement in the model results. The F-score is proportional to the ratio between precision and recall (see above in this document) and is directly proportional to the amount of correctly classified items. 
# 
# Overall, we conclude that the optimisation improves the model performances.
# 
# With respect to the naive benchmark analysed at question 1, the current models shows significant improvements. For the fomer, I have quantified an accuracy of 24.78% and a F-score of 29.17%. This shows that the analysis conducted in the present document has improved the ability to identyifies the donors (income higher than 50K) by a factor of 3.4.

# ----
# ## Feature Importance
# 
# An important task when performing supervised learning on a dataset like the census data we study here is determining which features provide the most predictive power. By focusing on the relationship between only a few crucial features and the target label we simplify our understanding of the phenomenon, which is most always a useful thing to do. In the case of this project, that means we wish to identify a small number of features that most strongly predict whether an individual makes at most or more than \$50,000.
# 
# Choose a scikit-learn classifier (e.g., adaboost, random forests) that has a `feature_importance_` attribute, which is a function that ranks the importance of features according to the chosen classifier.  In the next python cell fit this classifier to training set and use this attribute to determine the top 5 most important features for the census dataset.

# ### Question 6 - Feature Relevance Observation
# When **Exploring the Data**, it was shown there are thirteen available features for each individual on record in the census data. Of these thirteen records, which five features do you believe to be most important for prediction, and in what order would you rank them and why?

# **Answer:**
# 
# Classifying the income of a person based on predictive features is a challenging task as it may be influenced by personal behaviour, sibling attitudes, etc. Notwithstanding that, the aim is possible and it has been started in the analyses conducted herebefore. In particular, I have used the thirteen features described in the paragraph titled 'Feature set Exploration'. However, not all of them might be really necessary and, actually, some of them might be not correlated with the income, reducing the overall performace of the model. More in details, I believe that the five most relevant features for the current analysis are (1) occupation, (2) age, (3) education, (4)hours per week, and (5) sex:
# 
# - several reports and ranks across the globes have shown that salaries change significantly among occupations: for istance an handlers-cleaners earn an average of 12 dollars in the USA, while a tech-supporter gains 31 dollars. These results convert to 25K and 63K over the here, assuming that the employee works 40 hours every week.
# - Age is generally correlated with working experience and so it could be a key factor on the income estimate.
# - It is well documneted that people with a higher degree (e.g. master, doctorate) earn more than people without. For istance, gaining a doctorate degree could increase your salary by 15K at the most.
# - The correlation between amount of hours worked and salary is purely mathematical.
# - Reports also show that men earn on average 8K dollars more than women per year.
# 
# *Reference materials:*
# 
# www.indeed.com/salaries/Cleaner-Salaries
# 
# www.glassdoor.co.uk/Salaries/us-technical-support
# 
# www.wikipedia.org
# 
# https://oedb.org/advice/masters-vs-phd-degrees-for-graduate-students/

# ### Implementation - Extracting Feature Importance
# Choose a `scikit-learn` supervised learning algorithm that has a `feature_importance_` attribute availble for it. This attribute is a function that ranks the importance of each feature when making predictions based on the chosen algorithm.
# 
# In the code cell below, you will need to implement the following:
#  - Import a supervised learning model from sklearn if it is different from the three used earlier.
#  - Train the supervised model on the entire training set.
#  - Extract the feature importances using `'.feature_importances_'`.

# In[12]:


# Import a supervised learning model that has 'feature_importances_'

from sklearn.ensemble import AdaBoostClassifier

# Train the supervised model on the training set using .fit(X_train, y_train)
clf = AdaBoostClassifier(random_state = 1)
model = clf.fit(X_train, y_train)

# Extract the feature importances using .feature_importances_ 
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)


# ### Question 7 - Extracting Feature Importance
# 
# Observe the visualization created above which displays the five most relevant features for predicting if an individual makes at most or above \$50,000.  
# * How do these five features compare to the five features you discussed in **Question 6**?
# * If you were close to the same answer, how does this visualization confirm your thoughts? 
# * If you were not close, why do you think these features are more relevant?

# **Answer:**
# 
# The analysis conducted on the features importance reveals that age, education, and amount of hours worked per week are strictly correlated with the personal income. This result is expected and supported by the reasoning discussed in the previous answer. 
# 
# Surprisingly, however, nor the person sex or the specific occupation lie among the five most correlated features. The amount of capital loss and gain seem to better describe the variance in the predicted variable. This might be supported by the fact that higher incomes allow people to risk more and to support higher changes in their capital. 

# ### Feature Selection
# How does a model perform if we only use a subset of all the available features in the data? With less features required to train, the expectation is that training and prediction time is much lower — at the cost of performance metrics. From the visualization above, we see that the top five most important features contribute more than half of the importance of **all** features present in the data. This hints that we can attempt to *reduce the feature space* and simplify the information required for the model to learn. The code cell below will use the same optimized model you found earlier, and train it on the same training set *with only the top five important features*. 

# In[13]:


# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))


# ### Question 8 - Effects of Feature Selection
# 
# * How does the final model's F-score and accuracy score on the reduced data using only five features compare to those same scores when all features are used?
# * If training time was a factor, would you consider using the reduced data as your training set?

# **Answer:**
# 
# The use of the five most correlated features instead of the entire dataset reduces the Accuracy and F-score of the model by 5% and 20% respectively. However, if the training time is an issue (the Support Vector Classifier) I would consider the use of the reduced amount of fields as the final accuracy is still acceptable.

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
