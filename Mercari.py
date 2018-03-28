import csv

with open('train.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')

    for row in readCSV:
        print(row)

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import f1_score
score = f1_score(y, y_pred, average='macro')

y=df.E
X=df.drop(['E','G','I'],axis=1)

train_X,test_X,train_y,test_y=train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=230, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)

Test_df=pd.read_csv('./data/test.csv')
Test_X=Test_df.drop(['Ob','T6'],axis=1)
predictions = my_model.predict(Test_X.as_matrix())

print(predictions)

my_submission = pd.DataFrame({'item_id': Test_df.Ob, 'category_class': predictions})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False,columns=['item_id','category_class'])

