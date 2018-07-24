
# coding: utf-8

# In[1]:

from googletrans import Translator
from tools import GetFar
from pprint import pprint
from operator import itemgetter
from predict.predict import *
import jieba

translator = Translator(service_urls=[
      'translate.google.com',
      'translate.google.com.tw/',
    ])


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

# In[2]:

def segment(sent):
    seg_list = jieba.cut(sent)
    return [seg for seg in seg_list if seg.strip()]


def format_data(data):
    return {
        'conversation': data['conversation'].split('\n'),
        'conversation_pin': data['conversation'].split('\n'),
        'question': data['question'],
        'question_pin': data['question'],
        'options': data['options'].split('\n'),
        'options_pin': data['options'].split('\n')
    }


def get_content(line):
    return line.split('\t')[1] or '<NONE>'


def to_passage(conversation):
    lines = [get_content(line) for line in conversation]
    return '。'.join(lines)
    

def get_opt_pair(options):
    for opt in options:
        yield opt.split('\t')[0], get_content(opt)


# In[5]:

def main_process(datas):
    datas = [format_data(data) for data in datas]

    model_inputs = []
    for data in datas:
        passage = to_passage(data['conversation'])
        question = get_content(data['question'])
        options = [ (i, opt) for i, opt in get_opt_pair(data['options']) ]
        
        content_opts = list(map(lambda x: x[1], options))
        query = translator.translate('\n'.join( [passage, question, '\n'.join(content_opts)] ), dest='en', src='zh-TW')
        
        query = query.text.split('\n')
        model_inputs.append( {
            'passage': query[0],
            'question': query[1],
            'options': [(pair[0], q) for pair, q in zip(options, query[2:])]
        } )
    results = predict_batch_json(model_inputs)
    pprint(results)

    answers = []
    for result in results:
        cosine_pair = result['cosine'].items()
        ans_idx = max(cosine_pair, key=itemgetter(1))[0]
        scores = list(map(lambda pair: pair[1], cosine_pair))
        answers.append({'answer': ans_idx, 'scores': scores})

    # First method - 反向指標
    # ans_idx, scores = GetFar(conver, opt_list)
    
    return answers   


# In[6]:

# test
if __name__ == "__main__":
    test = [{
      "conversation": "1\t我剛吃飽來散步\t0.99\n2\t你吃飽了嗎\t0.33\n1\t吃飽囉\t0.87",
      "conversation_pin": "1\t我剛吃飽來散步\t0.99\n2\t你吃飽了嗎\t0.33\n1\t吃飽囉\t0.87",
      "question": "1\t請問他吃飽了嗎\t0.66",
      "question_pin": "1\t請問他吃飽了嗎\t0.66",
      "options": "1\t吃飽了\t0.99\n2\t還沒吃\t0.99",
      "options_pin": "1\t吃飽了\t0.99\n2\t還沒吃\t0.99"
    }]
    print(main_process(test))


# In[ ]:

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)


@app.route('/', methods=['POST'])
def answer():
    # logging.debug('Get request')
    topic = request.get_json()
    # logging.debug(topic)
    
    if not topic: return jsonify({'status': 'wrong'})
    
    answers = main_process(topic)

    return jsonify(answers)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1315)


# In[ ]:



