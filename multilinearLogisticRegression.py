import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
import pandas as pd

df = pd.read_csv('data_classification.csv')

hours_slept = df['Hours_Slept'].tolist()

hours_studied = df['Hours_studied'].tolist()

fig = px.scatter(x = hours_slept, y = hours_studied)

fig.show()

hours_slept = df['Hours_Slept'].tolist()

hours_studied = df['Hours_studied'].tolist()

results = df['results'].tolist()

colors = []

for data in results:
  if data == 1:
    colors.append('green')
  else:
    colors.append('red')

fig = go.Figure(data = go.Scatter(
    x = hours_studied,
    y = hours_slept,
    mode = 'markers',
    marker = dict(color = colors)
))

fig.show()

hours = df[['Hours_studied', 'Hours_Slept']]

results = df['results']

hours_train, hours_test, results_train, results_test = train_test_split(hours, results, test_size = 0.25, random_state = 0)

print(hours_train)

classifier = LogisticRegression(random_state = 0) 

classifier.fit(hours_train, results_train)

results_pred = classifier.predict(hours_test)

print ("Accuracy : ", accuracy_score(results_test, results_pred))

user_hours_studied = int(input('Enter the hours studied:'))

user_hours_slept = int(input('Enter the hours slept'))

user_test = sc_x.transform([[user_hours_studied, user_hours_slept]])

user_results_pred = classifier.predict(user_test)

if user_results_pred[0] == 1:
  print('The user may pass.')
else:
  print('The user may not pass.')