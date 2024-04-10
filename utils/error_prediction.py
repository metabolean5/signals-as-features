# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import pandas as pd
from matplotlib import pyplot as plt
import joblib
import pprint as pp
# load data


with open("relations_features_highlevel_train_test.json",'r') as file:
    relations_features_train_test = json.load(file)

df = pd.DataFrame.from_dict(relations_features_train_test, orient='index')

features_names = df.columns.tolist()[1:]
print(features_names)

row_names = df.index
dataset = df.values

print(dataset)

# split data into X and y
Y = dataset[:, 0]
X = dataset[:, 1:]
# split data into train and test sets

print('setting up train test split...')
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test, row_names_train, row_names_test = train_test_split(
    X, Y,row_names, test_size=test_size, random_state=seed
)

# fit model no training data
print('building model...')


model = XGBClassifier() 
model.fit(X_train, y_train)
joblib.dump(model, 'XGBClassifier_highlevel.joblib')


#model = joblib.load('XGBClassifier_highlevel.joblib')


print(model.feature_importances_)
input()
feature_importances = model.feature_importances_.tolist()
# plot
sorted_indices = sorted(range(len(feature_importances)), key=lambda k: feature_importances[k], reverse=True)

sorted_labels = [features_names[i] for i in sorted_indices][:23]
sorted_importances = [feature_importances[i] for i in sorted_indices][0:23]


# Plotting
fig, ax = plt.subplots(figsize=(2, 4))
ax.barh(range(len(sorted_importances)), sorted_importances, align='center')

plt.yticks(range(len(sorted_importances)), sorted_labels, fontsize=10)  # Adjust the font size as needed


# Set the y-axis labels with sorted feature names
plt.yticks(range(len(sorted_importances)), sorted_labels)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances (Ranked)')


# Save the plot as an image file (e.g., PNG)
plt.savefig('feature_importances_plot_horizontal_ranked.png', bbox_inches='tight')






# make predictions for test data
print('making predictions...')
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

print(y_test)
print(predictions)
print(row_names_test.tolist())
# evaluate predictions
accuracy = accuracy_score(y_test.tolist(), predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


with open("rst_relations.json",'r') as file:
    rst_relations = json.load(file)

with open("relation2signal.json",'r') as file:
    relation2signal = json.load(file)

   

relations_test = row_names_test.tolist()
relations_all = row_names.tolist()
header = ['file','orginal_prediction', 'error_analysis_prediction','relation','predicted_relation','edus','signals']

data_unsure = []


all = 0
for i,r in enumerate(relations_all):
    print(r)

    error_analysis_prediction = 0
    

    if r in rst_relations['error']:
        orginal_prediction = 'error'
        predicted_relation = rst_relations[orginal_prediction][r]['relation_pred']
        relation = rst_relations[orginal_prediction][r]['relation_gold']
    else:
        orginal_prediction = 'correct'
        relation = rst_relations[orginal_prediction][r]['relation']
        predicted_relation = relation


    edus_text = rst_relations[orginal_prediction][r]['edus_text']

    fname = '_'.join(r.split('_')[:2])+'.txt'
    edus =  '_'.join(r.split('_')[2:4])

    print(relation2signal[fname][edus])

    if predicted_relation == 'enablement':
        all +=1

    if len(relation2signal[fname][edus]) <=0 :
        continue
    signals = relation2signal[fname][edus][0]['features']


    error_analysis_prediction = 1
    row = [r, orginal_prediction, error_analysis_prediction,relation,predicted_relation, edus_text,signals ]
    data_unsure.append(row)



print(all)
input()


df = pd.DataFrame(data_unsure, columns =header)

'''

csv_file_path = 'error_prediction_full.csv'
df.to_csv(csv_file_path, index=False)


feature_significance = {}
for feature in features_names:
    for index, row in df.iterrows():

        if row['error_analysis_prediction'] == 0:continue

        feature_significance.setdefault(feature,{'correct':0, 'error': 0, 'total':0})
        
        if feature in row['signals'] and row['orginal_prediction'] == 'correct':
            feature_significance[feature]['correct'] +=1
            feature_significance[feature]['total'] +=1
        elif feature in row['signals']:
            feature_significance[feature]['error'] +=1
            feature_significance[feature]['total'] +=1

#normalize
to_del = []
for feature in feature_significance:
    print(feature)

    nb_correct = feature_significance[feature]['correct']
    nb_error = feature_significance[feature]['error']

    if nb_error ==0 and nb_correct ==0: 
        to_del.append(feature)
        continue
        

    feature_significance[feature]['correct'] = round(nb_correct / (nb_correct+nb_error) ,2)
    feature_significance[feature]['error'] = round(nb_error / (nb_correct+nb_error),2)

for feature in to_del:
    del feature_significance[feature]


sorted_correct = sorted(feature_significance.items(), key=lambda x: x[1]['correct'], reverse=True)

sorted_correct = {key: value for key, value in sorted_correct}
pp.pprint(sorted_correct)


keys = list(sorted_correct.keys())
correct_values = [item['correct'] for item in sorted_correct.values()]

plt.figure(figsize=(10,10))

# Plot the horizontal bar chart
plt.barh(keys, correct_values, color='green')
plt.xlabel('Correct Values')
plt.title('Correct Values for Each Signal')
plt.show()
plt.yticks(fontsize=12)

plt.savefig('correct_important_signals_verified.png', bbox_inches='tight')

'''


relation_signals_significance = {}
all = 0
relation_num = {}
for index, row in df.iterrows():
    relation = row['relation']

    relation_num.setdefault(relation,0)
    relation_num[relation]+=1

    relation_signals_significance.setdefault(relation,{})

    for feature in features_names:

        if feature in row['signals']:
            relation_signals_significance[relation].setdefault(feature, {'correct':0, 'error': 0, 'total':0})

        if feature in row['signals'] and row['orginal_prediction']=='correct':
            relation_signals_significance[relation][feature]['correct']+=1
            relation_signals_significance[relation][feature]['total']+=1
        elif feature in row['signals']:
            relation_signals_significance[relation][feature]['error']+=1
            relation_signals_significance[relation][feature]['total']+=1

            if feature == 'dm' and relation == 'background':
                print(row)
                input()
    
    all+=1


pp.pprint(relation_num)
input()
    





for relation in relation_signals_significance:

    for feature in relation_signals_significance[relation]:
        print(feature)

        nb_correct = relation_signals_significance[relation][feature]['correct']
        nb_error = relation_signals_significance[relation][feature]['error']     

        relation_signals_significance[relation][feature]['correct'] = round(nb_correct / (nb_correct+nb_error) ,2)
        relation_signals_significance[relation][feature]['error'] = round(nb_error / (nb_correct+nb_error),2)
    




pp.pprint(relation_signals_significance)

df_test = pd.DataFrame(relation_signals_significance)

print(df)
