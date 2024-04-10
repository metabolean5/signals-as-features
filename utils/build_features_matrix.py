import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pprint as pp

def plot_confusion_matrix(true_labels,predicted_labels):
    # Example true labels and predicted labels
    unique_labels = list(set(true_labels + predicted_labels))
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)

    # Normalize the confusion matrix by dividing each cell by the sum of true labels for that class
    normalized_cm = cm / cm.sum(axis=1)[:, np.newaxis]

    # Plot the confusion matrix using seaborn with normalized values
    plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_cm, annot=True, fmt=".0%", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix_rst_french.png")
    print('Done')


def get_aligned_pred_gold(rst_relations):
    true_labels = []
    predicted_labels = [] 
    for truth in rst_relations:
        for relation in rst_relations[truth]:
            if truth == "correct":
                true_labels.append(rst_relations[truth][relation]['relation'])
                predicted_labels.append(rst_relations[truth][relation]['relation'])
            else:
                true_labels.append(rst_relations[truth][relation]['relation_gold'])
                predicted_labels.append(rst_relations[truth][relation]['relation_pred'])
    
    return true_labels, predicted_labels



def plot_distance_distrib(rst_relations):
    distance_distrib = {}
    for truth in rst_relations:
        for relation in rst_relations[truth]:
            distance = rst_relations[truth][relation]['N-distance']
            if distance > 13: distance = 14
            distance_distrib.setdefault(distance,0)
            distance_distrib[distance]+=1
    
    data = distance_distrib
    data = {k: v for k, v in data.items() if k <= 14}
    data = dict(sorted(data.items()))

    x = list(data.keys())
    y = list(data.values())

    pp.pprint([x,y])
    fig, ax = plt.subplots()
    ax.bar(x, y, color='royalblue')

    plt.title('Distance Distribution')
    plt.xlabel('Distance')
    plt.ylabel('Number of occurences')
    # Show or save the plot
    plt.savefig("distance_distrib.png")


def distance_stats(rst_relations):
    distance_stats = {}
    for truth in rst_relations:
        for relation in rst_relations[truth]:
            distance_num = rst_relations[truth][relation]['N-distance']
            if distance_num >= 13: distance = ">=13"
            if distance_num == 0: distance = "0"
            if 1 <= distance_num and distance_num <= 3: distance = "1-3"
            if 4 <= distance_num and distance_num <= 6: distance = "4-6"
            if 7 <= distance_num and distance_num <= 9: distance = "7-9"
            if 10 <= distance_num and distance_num <= 12: distance = "10-12"

            distance_stats.setdefault(distance,{'correct':0, 'error':0, 'occ':0,'relations':{}})
            distance_stats[distance]['occ']+=1

            if truth == "correct":
                distance_stats[distance]['correct']+=1
                r = rst_relations[truth][relation]['relation']
                distance_stats[distance]['relations'].setdefault(r,0)
                distance_stats[distance]['relations'][r]+=1
            else:
                distance_stats[distance]['error']+=1
                r = rst_relations[truth][relation]['relation_gold']
                distance_stats[distance]['relations'].setdefault(r,0)
                distance_stats[distance]['relations'][r]+=1
    
    distance_stats_acc = {}

    for ds in distance_stats:
        distance_stats_acc.setdefault(ds,{'Accuracy':0, 'Top relations': None})
        tp = distance_stats[ds]['correct']
        fp = distance_stats[ds]['error']
        acc = tp / (tp+fp)
        top_relations = sorted(distance_stats[ds]['relations'].items(), key=lambda x:x[1], reverse=True)
        first_3_relations = top_relations[:3]
        distance_stats_acc[ds]['Accuracy'] = str(acc)[:4]
        distance_stats_acc[ds]['Top relations'] = first_3_relations
    
    pp.pprint(distance_stats_acc)


def plot_distance_acc(rst_relations):
    distance_acc = {}
    for truth in rst_relations:
        for relation in rst_relations[truth]:
            distance = rst_relations[truth][relation]['N-distance']
            if distance > 13: distance = 14
            distance_acc.setdefault(distance,{'correct':0, 'error':0, 'occ':0})
            distance_acc[distance]['occ']+=1
            if truth == "correct":
                distance_acc[distance]['correct']+=1
            else:
                distance_acc[distance]['error']+=1

    data = distance_acc
    data = dict(sorted(data.items()))
    #data = {k: v for k, v in data.items() if k < 40}
    x = list(data.keys())
    
    y = []
    for d in data:
        y.append(data[d]["correct"]/(data[d]['correct']+data[d]['error']))

    pp.pprint([x,y])
    fig, ax = plt.subplots()
    ax.bar(x, y, color='royalblue')

    plt.title('Distance Accuracy')
    plt.xlabel('Distance')
    plt.ylabel('Accuracy')
    # Show or save the plot
    plt.savefig("distance_acc.png")
    

def get_gold_label_distrib(rst_relations):
    relation_gold_distib = {} 
    for truth in rst_relations:
        for relation in rst_relations[truth]:
            if truth ==  "correct":
                pp.pprint(rst_relations[truth][relation])
                relation_gold_distib.setdefault(rst_relations[truth][relation]['relation'],0)
                relation_gold_distib[rst_relations[truth][relation]['relation']] +=1
            else:
                pp.pprint(rst_relations[truth][relation])
                relation_gold_distib.setdefault(rst_relations[truth][relation]['relation_gold'],0)
                relation_gold_distib[rst_relations[truth][relation]['relation_gold']] +=1

    #relation_gold_distib = sorted(relation_gold_distib.items(), key=lambda x:x[1])
    
    rgd_percentqge = {}
    total_nb = sum(relation_gold_distib.values())
    print(total_nb)
    for rel in relation_gold_distib:
        rgd_percentqge.setdefault(rel,[None,None])
        rgd_percentqge[rel][0] = round(((relation_gold_distib[rel] / total_nb) * 100), 2)
        rgd_percentqge[rel][1] = relation_gold_distib[rel]
    rgd_percentqge = sorted(rgd_percentqge.items(), key=lambda x:x[1],reverse=True)
    pp.pprint(rgd_percentqge)


def dm_stats(rst_relations,relation2signal):
     
    dm_present = {"correct":0,"error":0} 
    for truth in rst_relations:
        for relation in rst_relations[truth]:
            file = '_'.join(relation.split('_')[0:2])+'.txt'
            edus = '_'.join(relation.split('_')[2:4])

            print(file)
            print(edus)
            for signal in relation2signal[file][edus]:
                if "dm;" in signal['features'] and truth == "correct":
                    dm_present['correct']+=1
                elif "dm;" in signal['features']:
                    dm_present['error']+=1
    
    pp.pprint(dm_present)



def intra_stats(rst_relations,relation2signal):

    is_intra = {"correct":0,"error":0} 
    for truth in rst_relations:
        for relation in rst_relations[truth]:
            
            if rst_relations[truth][relation]['N-distance'] == 0 and truth == 'correct':
                is_intra['correct']+=1
            elif rst_relations[truth][relation]['N-distance'] == 0 and truth == "error":
                is_intra['error']+=1
    
    pp.pprint(is_intra)


def signal_stats(rst_relations,relation2signal):

    signal_all = {}
    signals_stats = {"correct":{},"error":{}} 
    for truth in rst_relations:
        for relation in rst_relations[truth]:
            file = '_'.join(relation.split('_')[0:2])+'.txt'
            edus = '_'.join(relation.split('_')[2:4])

            print(file)
            print(edus)
            signals_set = []
            for signal in relation2signal[file][edus]:
                signals = signal['features'].split(';')
                for ss in signals:
                    signals_set.append(ss)
            signals_set = set(signals_set)
            for s in signals_set:
                if truth == "correct":
                    signals_stats['correct'].setdefault(s,0)
                    signals_stats['correct'][s]+=1
                    signal_all.setdefault(s)
                else:
                    signals_stats['error'].setdefault(s,0)
                    signals_stats['error'][s]+=1
                    signal_all.setdefault(s)
    
    

    for truth in signals_stats:
        sorted_signals = sorted(signals_stats[truth].items(), key=lambda x:x[1],reverse=True)
        pp.pprint(sorted_signals)
        pp.pprint(signal_all)



def write_files(rst_relations,relation2signal):


    for truth in rst_relations:
        for relation in rst_relations[truth]:

            file = '_'.join(relation.split('_')[0:2])
            edus = '_'.join(relation.split('_')[2:4])

            to_check = file+'\nPrediction for '+ edus


            if truth == "correct":
                file_path = "correct_signals.txt"
                relation_gold = rst_relations[truth][relation]['relation']
                relation_pred = relation_gold
            else:
                file_path = "error_signals.txt"
                relation_gold = rst_relations[truth][relation]['relation_gold']
                relation_pred = rst_relations[truth][relation]['relation_pred']

            with open(file_path, 'a') as file_n:
                file_n.write("\n")

                file_n.write(str(file)+" "+ str(edus)+ "\n")
                file_n.write('Prediction for ' + edus+' : ' + relation_pred+"\n")
                file_n.write('Gold Annotation for ' + edus+' : ' + relation_gold+"\n")
                file_n.write('gold EDUS:'+  rst_relations[truth][relation]['edus_text']+"\n")

                signals_all = []
                singal_file = file+'.txt'
                print(edus)
                for signal in relation2signal[singal_file][edus]:
                    signals = signal['features'].split(';')
                    for s in signals:
                        signals_all.append(s)
                    
                signals = set(signals_all)
                file_n.write('Signals ' + edus+' : ' +str(signals)+"\n")


def distance_singals_stats(rst_relations,relation2signal):

    explicit = ['dm','syntactic',
    "relative_clause",
    "infinitival_clause",
    "present_participial_clause",
    "past_participial_clause",
    "imperative_clause",
    "interrupted_matrix_clause",
    "parallel_syntactic_construction",
    "reported_speech",
    "subject_auxiliary_inversion",
    "nominal_modifier",
    "adjectival_modifier",
    "graphical"
]
    implicit = ["synonymy",
    "antonymy",
    "meronymy",
    "repetition",
    "indicative_word_pair",
    "lexical_chain",
    "general_word",'reference','lexical','semantic','morphological','genre','numerical','(lexical+syntactic)','(indicative_word+participial_clause)','(propositional_reference+subject_np)','(graphical+syntactic)','(comma+present_participial_clause)','(syntactic+semantic)','(parallel_syntactic_constructions+lexical_chain)','(lexical_chain+subject_np)',"(repetition+subject_np)","(semantic+syntactic)","personal_reference+subject_np","(reference+syntactic)"]


    distance_stats = {}
    for truth in rst_relations:
        for relation in rst_relations[truth]:
            distance_num = rst_relations[truth][relation]['N-distance']
            edu_text = rst_relations[truth][relation]["edus_text"]
            if distance_num >= 13: distance = ">=13"
            if distance_num == 0: distance = "0"
            if 1 <= distance_num and distance_num <= 3: distance = "1-3"
            if 4 <= distance_num and distance_num <= 6: distance = "4-6"
            if 7 <= distance_num and distance_num <= 9: distance = "7-9"
            if 10 <= distance_num and distance_num <= 12: distance = "10-12"

            distance_stats.setdefault(distance,{'correct':0, 'error':0, 'occ':0,'signals':{},'signals_nb':0})
            

            fname = '_'.join(relation.split('_')[:2])+'.txt'
            edus =  '_'.join(relation.split('_')[2:4])

            signals_list = relation2signal[fname][edus]

            final_signals = []
            continue_b = True
            for signals in signals_list:
                for s in signals['features'].split(';'):
                    if s in explicit:
                        continue_b = False
                        final_signals.append(s)
                    if s in implicit:
                        final_signals.append(s)

            
            if continue_b:continue

            if 'relation' not in rst_relations[truth][relation]:
                relation = rst_relations[truth][relation]['relation_pred']
            else:
                relation = rst_relations[truth][relation]['relation']

            #if relation != 'elaboration':
             #   continue

           # if '.]['  in edu_text:
        #      continue

            distance_stats[distance].setdefault('relations',{})
            distance_stats[distance]['relations'].setdefault(relation,0)
            distance_stats[distance]['relations'][relation]+=1

            
            

            distance_stats[distance]['occ']+=1
            if truth == "correct":
                distance_stats[distance].setdefault('correct',0)
                distance_stats[distance]['correct']+=1
            else:
                distance_stats[distance].setdefault('error',0)
                distance_stats[distance]['error']+=1
            
            for signal in final_signals:
                distance_stats[distance]['signals'].setdefault(signal,0)
                distance_stats[distance]['signals'][signal]+=1
                distance_stats[distance]['signals_nb']+=1

           
    
    distance_stats_acc = {}

    for ds in distance_stats:
        distance_stats_acc.setdefault(ds,{'Accuracy':0, 'Top signals': None,'Top relations': None, "Occ": distance_stats[ds]['occ'] })
        tp = distance_stats[ds]['correct']
        fp = distance_stats[ds]['error']
        if (tp+fp)==0:fp=1
        acc = tp / (tp+fp)
        top_signals = sorted(distance_stats[ds]['signals'].items(), key=lambda x:x[1], reverse=True)
        top_relations = sorted(distance_stats[ds]['relations'].items(), key=lambda x:x[1], reverse=True)
        first_5_relations = top_signals[:5]
        #first_5_relations = [(item[0], str(item[1] *100 / distance_stats[ds]['correct_nb']).split('.')[0]+'%' ) for item in first_5_relations]
        distance_stats_acc[ds]['Accuracy'] = str(acc)[:4]
        distance_stats_acc[ds]['Top signals'] = first_5_relations
        distance_stats_acc[ds]['Top relations'] = top_relations
    
    pp.pprint(distance_stats_acc)
    nb_dms_correct = []
    nb_dms_wrong = []


def build_feature_matrix(rst_relations,relation2signal, features):
    relations_features_train_test = {}

    for truth in rst_relations:
        for relation in rst_relations[truth]:
            fname = '_'.join(relation.split('_')[:2])+'.txt'
            edus =  '_'.join(relation.split('_')[2:4])

            relations_features_train_test.setdefault(relation,{})

            if truth=="correct":
                y = 0
            else:
                y = 1
            relations_features_train_test[relation].setdefault('Error',y)

            signals_list = relation2signal[fname][edus]
            final_signals = []
            continue_b = True
            for signals in signals_list:
                for s in signals['features'].split(';'):
                    if s in features:
                        final_signals.append(s)
                        relations_features_train_test[relation].setdefault(s,True)
            
    return relations_features_train_test
                
            
features = [
    '(comma+present_participial_clause)',
    '(general_word+subject_np)',
    '(indicative_word+participial_clause)',
    '(lexical_chain+subject_np)',
    '(meronymy+subject_np)',
    '(parallel_syntactic_constructions+lexical_chain)',
    '(personal_reference+subject_np)',
    '(present_participial_clause+beginning)',
    '(propositional_reference+subject_np)',
    '(repetition+subject_np)',
    '(synonymy+subject_np)',
    'adj_modifier',
    'alternative_expression',
    'antonymy',
    'colon',
    'combined',
    'comparative_reference',
    'dash',
    'demonstrative_reference',
    'dm',
    'imperative_clause',
    'indicative_word',
    'indicative_word_pair',
    'infinitival_clause',
    'interrupted_matrix_clause',
    'inverted_pyramid_scheme',
    'items_in_sequence',
    'lexical_chain',
    'meronymy',
    'newspaper_layout',
    'newspaper_style_attribution',
    'nominal_modifier',
    'numerical',
    'parallel_syntactic_constructions',
    'parentheses',
    'past_participial_clause',
    'personal_reference',
    'present_participial_clause',
    'propositional_reference',
    'reference',
    'relative_clause',
    'repetition',
    'reported_speech',
    'same_count',
    'semi-colon',
    'synonymy',
    'tense',
    'unsure',
]


features_highlevel = {'dm', 'reference', 'lexical', 'syntactic', 'semantic', 'morphological', 'graphical',
                      'genre', 'numerical', '(reference+syntactic)', '(semantic+syntactic)', '(lexical+syntactic)'
                      , '(syntactic+semantic)', '(syntactic+positional)','(graphical+syntactic)', 'unsure' }
              
with open("rst_relations_french.json",'r') as file:
    rst_relations = json.load(file)

with open("relation2signal.json",'r') as file:
    relation2signal = json.load(file)

'''

relations_features_train_test = build_feature_matrix(rst_relations,relation2signal, features_highlevel)

with open("relations_features_highlevel_train_test.json",'w') as file:
    json.dump(relations_features_train_test,file)


pp.pprint(relations_features_train_test)
'''



true_labels , predicted_labels = get_aligned_pred_gold(rst_relations)
plot_confusion_matrix(true_labels, predicted_labels)

'''

write_files(rst_relations,relation2signal)
signal_stats(rst_relations,relation2signal)
distance_stats(rst_relations)
distance_singals_stats(rst_relations,relation2signal)
'''




