import json
import pprint as pp
import glob
import xml.etree.ElementTree as ET

def parse_xml(xml_file):
    # Load the XML file
    tree = ET.parse(xml_file) 
    root = tree.getroot()
    segments = root.findall(".//segments/segment")
    segment_list = []

    for segment in segments:
        segment_id = segment.get('id')
        start = segment.get('start')
        end = segment.get('end')
        features = segment.get('features')
        state = segment.get('state')
        comment = segment.get('comment') if 'comment' in segment.attrib else None

        segment_data = {
            'id': segment_id,
            'start': start,
            'end': end,
            'features': features,
            'state': state,
            'comment': comment
        }

        segment_list.append(segment_data)

    return segment_list

def get_rst_signals():

    pattern = 'RST_signals_xml/**'  

    matching_paths_xml = glob.glob(pattern, recursive=True)
    matching_paths_gold_trees = glob.glob("gold_RST_trees/*.txt", recursive=True)

    gold_trees = {}
    for path in matching_paths_gold_trees:
        gold_trees.setdefault(path.split('/')[1])

    rst_signals = {}
    for path in matching_paths_xml:
        if "Signal.xml" in path and path.split('/')[1] in gold_trees:
            segments = parse_xml(path)
            rst_signals.setdefault(path.split('/')[1],segments)
            
    return rst_signals

def get_relation_line(file,span):
    line_number = 0  # Initialize line number variable
    # Open the file in read mode
    with open(file, 'r') as file:
        for line in file:
            line_number += 1
            if span in line:
                #print(f"String '{span}' found on line {line_number}:")
                #print(line.strip())  # Print the entire line where the string is found
                return line_number  
        
def get_gtr2line():

    with open("gold_trees_dic.json", "r") as file:
        gold_trees = json.load(file)

    gtr2line = {} # gold tree relation to line
    for file in gold_trees:
        #print(file)
        for rel in gold_trees[file]:

            if  gold_trees[file][rel][1][0] == 'N':
                if gold_trees[file][rel][1] == 'N-N': 
                    span_txt = list(map(str, gold_trees[file][rel][0]['seg2span']))
                else:
                    span_txt = list(map(str, gold_trees[file][rel][0]['seg2span']))
                span = "("+ str(' '.join( span_txt)) +')'
            else:
                span_txt = list(map(str, gold_trees[file][rel][0]['seg1span']))
                span = "("+' '.join( span_txt)+')'
 
            gtr2line.setdefault(file.split('/')[1],{})
            gtr2line[file.split('/')[1]].setdefault(get_relation_line(file,span),rel)
    
    with open('gtr2line.json', "w") as file:
        json.dump(gtr2line, file, indent=4)

    return gtr2line

def get_signal_line(file_path,start_position):
    line_number = 1
    current_position = 0
    decalage = 0
    with open(file_path, 'r') as file:
        file_text = file.read()
    # Open the file in read mode
    with open(file_path, 'r') as file:
        for line in file:
            line_length = len(line)
            current_position += line_length
            decalage +=1
            # Check if the position falls within this line
            if current_position >= start_position - decalage >= (current_position - line_length):
                print(f"Character at position {start_position} is in line {line_number}:")
                print(line.strip())
                break

            line_number += 1

    return line_number


def get_signal2line():

    rst_signals = get_rst_signals()
    none_nb = 0
    signal2line = {}
    for file in rst_signals:
        #print(file)
        signal2line.setdefault(file,{})
        for signal in rst_signals[file]:
            start = int(signal['start'])
            end = int(signal['end'])
            fpath = "gold_RST_trees/"+str(file)
            line_nb = get_signal_line(fpath,start)

            signal2line[file].setdefault(line_nb,[])
            signal2line[file][line_nb].append(signal)
            
    #pp.pprint(signal2line)
    
    with open('signal2line.json', "w") as file:
        json.dump(signal2line, file, indent=4)

    return signal2line


def get_relation2singal():
    with open("gold_trees_dic.json", "r") as file:
        gold_trees = json.load(file)
    relation2signal = {}
    signal2line = get_signal2line()
    gtr2line = get_gtr2line()

    for file in gtr2line:
        relation2signal.setdefault(file,{})
        for rel in gtr2line[file]:
            line_nb = rel
            edus = gtr2line[file][rel]
            if line_nb not in signal2line[file]:
                file_path = "gold_RST_trees/" + file
                print(file)
                print(gtr2line[file][rel])
                print(gold_trees[file_path][gtr2line[file][rel]])
                input()
                relation2signal[file].setdefault(edus.replace(' ',''),[])
                continue
            relation2signal[file].setdefault(edus.replace(' ',''),signal2line[file][line_nb])

    pp.pprint(relation2signal)
    with open('relation2signal.json', "w") as file:
        json.dump(relation2signal, file, indent=4)
    return relation2signal



#get_relation2singal()



#getting examples

with open("relation2signal.json", "r") as file:
    relation2signal = json.load(file)

with open("rst_relations_french.json", "r") as file:
    rst_relations = json.load(file)


correct = 0
error = 0 
que = 0


for doc in relation2signal:
    for rel in relation2signal[doc]:
        
        for signal in relation2signal[doc][rel]:
            if ';' in signal['features']:
                rel_key = doc.replace('.txt', '') + '_'+rel
                if rel_key in rst_relations['correct']:

                    if rst_relations['correct'][rel_key]['relation'] == 'background' and' que ' in rst_relations['correct'][rel_key]['edus_text']:
                        print(doc)
                        print(rel)
                        print(signal)
                        print(rst_relations['correct'][rel_key])
                        input()
                        correct+=1
                                              
                elif rel_key in rst_relations['error']:
                
                    if rst_relations['error'][rel_key]['relation_gold'] == 'background' :

                        if ' que ' in rst_relations['error'][rel_key]['edus_text']:



                            print(doc)
                            print(rel)
                            print(signal)
                            print(rst_relations['error'][rel_key])
                            input()
                           
                            error+=1
                         
                    
        


print(correct)
print(error)
print()
print(que)








#get_relation2singal()

