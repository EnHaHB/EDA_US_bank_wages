## import libraries
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

# read data

wages = pd.read_table("us_bank_wages/us_bank_wages.txt", header=0,index_col=0)
wages.columns = [c.lower() for c in wages.columns]
# get rid of numbers - use actual info
wages['gender'].replace({0: 'Female', 1: 'Male'}, inplace=True)
wages['minority'].replace({0: 'Non-minority', 1: 'Minority'},inplace=True)
wages['jobcat'].replace({1: 'Administration', 2: 'Custodian', 3: 'Management'},inplace=True)

# define categorical variables as such
wages.educ.astype('category')
wages.gender.astype('category')
wages.minority.astype('category')
wages.jobcat.astype('category')

# create dummies for the categorical variables
educ_dummies = pd.get_dummies(wages['educ'], prefix='educ', drop_first=True)
gender_dummies = pd.get_dummies(wages['gender'], prefix='gender', drop_first=True)
minority_dummies = pd.get_dummies(wages['minority'], prefix='minority', drop_first=True)
jobcat_dummies = pd.get_dummies(wages['jobcat'], prefix='jobcat', drop_first=True)

# combine newly created dummies with the data set
# (and get rid off the equivalent non-dummies)
wages_reg = wages.drop(['educ', 'gender','minority','jobcat'], axis=1)
wages_reg = pd.concat([wages_reg, educ_dummies, gender_dummies, minority_dummies, jobcat_dummies], axis=1)
wages_reg.head()

# I used backwards elimination to select the model 
# final model based  on the selection via RMSE (see below)
X_1 = wages_reg.drop(['salary','educ_14', 'educ_20', 'educ_21'], axis=1)
#X = wages_reg[['salbegin', 'educ','gender','minority','jobcat']]
X_1 = sm.add_constant(X_1)
y_1 = wages_reg.salary

# split data set in train and test data set and model on the train set
X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(X_1,y_1, random_state= 42, shuffle=True, stratify=None)
model_1 = sm.OLS(y_1_train, X_1_train).fit()

adjr_model_1 = model_1.rsquared_adj
model_1.summary()

# predict the outcome of the model using the test and the training set
y_pred_1_train = model_1.predict(X_1_train)
y_pred_1_test = model_1.predict(X_1_test)

# calculate the MSE of each outcome
train_rmse_1= mean_squared_error(y_1_train, y_pred_1_train,squared=False)
test_rmse_1 = mean_squared_error(y_1_test, y_pred_1_test,squared=False)

#Take the square root of mse to find rmse
print('Train rmse: ', train_rmse_1)
print('Test rmse: ', test_rmse_1)

# calculate the difference of the test and training RMSE 
# to get info about the performance of the model
# i.e. how much off are the calculations in the train data
# compared to the test data

print (abs (train_rmse_1-test_rmse_1))

# save the model to disk
filename = 'wages_model.sav'
pickle.dump(model_1, open(filename, 'wb'))