import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt

df=pd.read_csv('fake-profile-dataset.csv')
df.dropna(inplace = True)

df = df.drop('followertofollowing',axis=1)

#split to train and test
y=df['isFake']
x=df.drop(['isFake'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=0)

#data normalization
scale = StandardScaler()
x_test = scale.fit_transform(x_test)
x_train = scale.fit_transform(x_train)

rfc = RandomForestClassifier(n_estimators=10000, random_state=0)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
print("RandomForest's Accuracy: ", metrics.accuracy_score(y_test,y_pred))

#find feature importance
feature_importance={}
for feat, importance in zip(df.columns, rfc.feature_importances_):
    feature_importance[feat]=importance
    print('feature: {f}, importance: {i}'.format(f=feat, i=importance))

names = list(feature_importance.keys())
values = list(feature_importance.values())

plt.bar(range(len(feature_importance)), values)
plt.xticks(range(len(feature_importance)), names, rotation=90)
plt.title("Feature Importance")
plt.show()

sfm = SelectFromModel(rfc, threshold=0.01)
sfm.fit(x_train, y_train)

selected_feature=[]
for feature_list_index in sfm.get_support(indices=True):
    selected_feature.append(df.columns[feature_list_index])
selected_feature.append('isFake')
for col in df.columns:
    if col not in selected_feature:
        df = df.drop(col, axis=1)
print(df.columns)
df.to_csv('thresh05.csv',index=False)
