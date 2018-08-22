
# coding: utf-8

# In[159]:

# from tools import GetFar
from googletrans import Translator
from opennmt_pinyin import pinyin2sentence
from predict.predict import *
from operator import itemgetter
import time
# import jieba
import logging, sys


# In[169]:

logger = logging.getLogger()
fh = logging.FileHandler('app.log')
fh.setLevel(logging.ERROR) #设置输出到文件最低日志级别
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') #定义日志输出格式
fh.setFormatter(formatter)
logger.addHandler(fh)


# ```
# {
#   'conversation': “1\t你好嗎\t0.926\n2\t我很好喔\n",
#   'conversation_pin’: “1\tni hat ma\t0.926\n2\twou hen hao\n",
#   'question’: “1\t請問…\t0.899“,
#   'question_pin’: “1\tchin woun …\t0.899“,
#   'options’: “1\t好\t0.99\n2\t不好0.933”
#   'options_pin’: “1\thao\t0.99\n2\tbu hao0.933”
# }
# 
# 
# { 
#   'answer': 2, 
#   'scores': [2, 1]
# }
# 
# 
# {  
#    "passage":"A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable of launching a payload into space more than once. This contrasts with expendable launch systems, where each launch vehicle is launched once and then discarded. No completely reusable orbital launch system has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were reused after several months of refitting work for each launch. The external tank was discarded after each flight.",
#    "question":"How many partially reusable launch systems were developed?"
# }
# ```

# In[161]:

FIELDS = ['conversation', 'conversation_pin', 'question', 'question_pin', 'options', 'options_pin']

def split_lines(data):
    return { field: data[field].split('\n') if field in data else[''] for field in FIELDS}


def compare(line, line_pin):
    line = line.strip().split('\t')
    line_pin = line_pin.strip().split('\t')
    
    if len(line) < 3 or not line[1].strip():
        return 'NULL'
    
    try:
        if float(line[2].strip()) > 0.1:
            return line[1]
        else:
            if not line_pin[1]: return 'NULL'

            scores, content = pinyin2sentence([line_pin[1]])
            return content[0][0]

    except Exception as e:
        logger.error(e)
        logger.error(line)
        logger.error(line_pin)
        return 'NULL'


def to_one_line(lines, lines_pin):
    lines = [compare(line, line_pin) for line, line_pin in zip(lines, lines_pin)]
    return '。'.join(lines)
    
    
def get_opt_pair(options, options_pin):
    for opt, opt_pin in zip(options, options_pin):
        yield opt.split('\t')[0], compare(opt, opt_pin)


def squeeze_entry(datas):
    entries = []
    options_mem = []
    for data in datas:
        passage = to_one_line(data['conversation'], data['conversation_pin'])
        question = to_one_line(data['question'], data['question_pin'])

        # get option number and content, then extract content part
        options = [ (i, opt) for i, opt in get_opt_pair(data['options'], data['options_pin']) ]    
        content_opts = list(map(lambda x: x[1], options))
    
        # Use \n to split tranlated sentence
        entries.append('\n'.join( [passage, question, '\n'.join(content_opts)] ).strip())
        options_mem.append(options)
    return entries, options_mem


def translate(datas):
    num = 100
    queries = []
    for i in range(0, 1500, num):
        translator = Translator(service_urls=[
              'translate.google.com',
              'translate.google.com.tw/',
              'translate.google.com.hk/',
              'translate.google.com.sg/',
              'translate.google.com.eg/',
              'translate.google.com.mx/'
        ])
        
        try:
            queries.extend(translator.translate(datas[i:i + num], dest='en'))
            
        except Exception as e:
            logger.error(e)
            queries.extend(['WRONG'])

    return queries


def construct(queries, optionss):
    model_inputs = []
    for i, query in enumerate(queries):
        query = query.text.split('\n')
        try:
            model_inputs.append( {
                'passage': query[0],
                'question': query[1],
                'options': [(pair[0], q) for pair, q in zip(optionss[i], query[2:])]
            } )
        except Exception as e:
            logger.error(e)
            logger.error(query)
            logger.error(optionss[i])
            model_inputs.append( {
                'passage': 'NONE',
                'question': 'NONE',
                'options': [(j+1, 'NONE') for j in range(4)]
            } )
            
    return model_inputs


# In[162]:

import pickle
    
def store_pkl(data):
    with open('tmp.pk', 'wb') as ws:
        pickle.dump(data, ws)
    print("FINISHED")

def read_pkl():
    with open('tmp.pk', 'rb') as fs:
        return pickle.load(fs)


# In[166]:

def main_process(datas):
    logger.critical("[To] split lines")
    datas = [split_lines(data) for data in datas]
    logger.critical("[Done] split lines")

    need_to_translate, optionss = squeeze_entry(datas)

    logger.critical('[To] translate')
    queries = translate(need_to_translate)
    logger.critical('[Done] translated')
    
    model_inputs = construct(queries, optionss)
    
    logger.critical('[To] input model')
    results = predict_batch_json(model_inputs)
    logger.critical('[Done] modeled')
    
    
    answers = []
    for result in results:
        cosine_pair = result['cosine'].items()
        if not cosine_pair:
            answers.append({'answer': 1, 'scores': [0, 0, 0, 0]})
            continue

        ans_idx = max(cosine_pair, key=itemgetter(1))[0]
        scores = list(map(lambda pair: pair[1], cosine_pair))
        answers.append({'answer': ans_idx, 'scores': scores})

    return answers   


# In[170]:

if __name__ == "__main__":
    import os

    A = sorted(['text/A/' + f for f in os.listdir('text/A') if f.endswith('.cm')])
    B = sorted(['text/B/' + f for f in os.listdir('text/B') if f.endswith('.cm')])
    C = sorted(['text/C/' + f for f in os.listdir('text/C') if f.endswith('.cm')])
    A_pin = sorted(['text/A/' + f for f in os.listdir('text/A') if f.endswith('.cm.syl')])
    B_pin = sorted(['text/B/' + f for f in os.listdir('text/B') if f.endswith('.cm.syl')])
    C_pin = sorted(['text/C/' + f for f in os.listdir('text/C') if f.endswith('.cm.syl')])

    bla = []
    for c, q, o, cp, qp, op in zip(A,B,C, A_pin, B_pin, C_pin):
        conversation = open(c).read().strip()
        question = open(q).read().strip()
        options = open(o).read().strip()
        conversation_p = open(cp).read().strip()
        question_p = open(qp).read().strip()
        options_p = open(op).read().strip()

        bla.append({
            'conversation': conversation,
            'conversation_pin': conversation_p,
            'question': question,
            'question_pin': question_p,
            'options': options,
            'options_pin': options_p
        })
        
    main_process(bla)


# In[ ]:


from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)


@app.route('/', methods=['POST'])
def answer():
    topic = request.get_json()
    
    if not topic: return jsonify({'status': 'wrong'})
    
    answers = main_process(topic)

    return jsonify(answers)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1315)

