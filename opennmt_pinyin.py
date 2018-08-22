# first load build_translator, please set the right path
# next commit the line 227 and 228 on onmt/translate/translator.py
#https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/translator.py#L227
from onmt.translate.translator import build_translator
import argparse
from uuid import uuid4
from typing import List

# these are values that you can change
# set properly on model_path, it is on google drive
# N_BEST is the highest score according to pinyin, not on a language model
GPUID = 0
MODEL_PT = 'model/pinyin2word_step_200000.pt'
BATCH_SIZE = 640
N_BEST = 1

opt = argparse.Namespace(
                   alpha=0.0,
                   attn_debug=False, 
                   batch_size=64, 
                   beam_size=5, 
                   beta=-0.0, 
                   block_ngram_repeat=0, 
                   coverage_penalty='none', 
                   data_type='text', 
                   dump_beam='', 
                   dynamic_dict=False, 
                   fast=False, 
                   gpu=GPUID, 
                   ignore_when_blocking=[], 
                   length_penalty='none', 
                   log_file='', 
                   max_length=100,
                   max_sent_length=None, 
                   min_length=0, 
                   model=MODEL_PT, 
                   n_best=N_BEST, 
                   output='model/tmp.txt', 
                   replace_unk=True, 
                   report_bleu=False, 
                   report_rouge=False, 
                   sample_rate=16000, 
                   share_vocab=False, 
                   stepwise_penalty=False, 
                   #tgt='data/word_test_tgt', 
                   verbose=False, 
                   window='hamming', 
                   window_size=0.02, 
                   window_stride=0.01)

#takes some time to build
print("building translator")
translator = build_translator(opt, report_score=False)

def pinyin2sentence(contents: List[str]):
    
    tempfile = "/tmp/pinyin%s" %(uuid4())
    with open(tempfile, "w") as output_file:
        for s in contents:
            print(s, file=output_file)
    
    #make predictions
    all_scores, all_predictions = translator.translate(
                    src_path=tempfile,
                    src_dir='',
                    batch_size=opt.batch_size,
                    attn_debug=opt.attn_debug)
    
    
    #convert score from tensor to float
    all_scores = [[float(score.cpu().numpy()) for score in scores] for scores in all_scores]
    return all_scores, all_predictions
