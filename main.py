
# coding: utf-8

# In[18]:

from GoogleTranslator import gtrans
from tools import GetFar
from pprint import pprint
from operator import itemgetter
import jieba
jieba.dt.cache_file = './jieba.cache.new'


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

# In[25]:

from predict.predict import *

def segment(sent):
    seg_list = jieba.cut(sent)
    return [seg for seg in seg_list if seg.strip()]


def format_data(data):
    return {
        'conversation': data['conversation'].strip().split('\n'),
        'conversation_pin': data['conversation'].strip().split('\n'),
        'question': data['question'].strip(),
        'question_pin': data['question'].strip(),
        'options': data['options'].strip().split('\n'),
        'options_pin': data['options'].strip().split('\n')
    }


def get_content(line):
    return line.split('\t')[1]


def to_passage(conversation):
    lines = [get_content(line) for line in conversation]
    return '。'.join(lines)
    

def get_opt_pair(options):
    return [(opt.split('\t')[0], get_content(opt)) for opt in options]

    
def main_process(data):
    data = format_data(data)
    
    # for mictsai
    model_input = {
        'passage': gtrans( to_passage(data['conversation']) ),
        'question': gtrans( get_content(data['question']) ),
        'options': [(i, gtrans(opt)) for i, opt in get_opt_pair(data['options'])]
    }

    pprint(model_input)
    result = predict_json(model_input)
    pprint(result)

    cosine_pair = result['cosine'].items()
    ans_idx = max(cosine_pair, key=itemgetter(1))[0]
    scores = list(map(lambda pair: pair[1], cosine_pair))

    # First method - 反向指標
    # ans_idx, scores = GetFar(conver, opt_list)
    
    return ans_idx, scores


# In[26]:

# test
if __name__ == "__main__":
    test = {
      "conversation": "1\t我剛吃飽來散步\t0.99\n2\t你吃飽了嗎\t0.33\n1\t吃飽囉\t0.87",
      "conversation_pin": "1\t我剛吃飽來散步\t0.99\n2\t你吃飽了嗎\t0.33\n1\t吃飽囉\t0.87",
      "question": "1\t請問他吃飽了嗎\t0.66",
      "question_pin": "1\t請問他吃飽了嗎\t0.66",
      "options": "1\t吃飽了\t0.99\n2\t還沒吃\t0.99",
      "options_pin": "1\t吃飽了\t0.99\n2\t還沒吃\t0.99"
    }
    print(main_process(test))


# In[37]:

#!/usr/bin/env python


from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)


@app.route('/', methods=['POST'])
def answer():
    topic = request.get_json()
    
    if not topic: return jsonify({'status': 'wrong'})
    
    ans_idx, scores = main_process(topic)
    
    return jsonify({ 
        'answer': ans_idx,
        'scores': scores
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1314)

