from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import json
import tensorflow as tf
import tensorboard as tb
import re
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def read_jsonl(input_file):
    with tf.io.gfile.GFile(input_file, 'r') as f:
        return [json.loads(ln) for ln in f]


def preprocess(lines = None, title = None, test_flag=False):
    with open('./dataset/'+str(title[:title.index('.jsonl')])+'.txt', 'w') as f:
        answer_tempelate = {0: '0', 1: '1'}
        for line in lines:
            tid = line['id']
            text = line['text']['stem']
            text = convert_to_unicode(re.sub(r'[\n\r;@#!"()]|http[s]?://\S+|[^\w.]', ' ', text))
            trigger_words = line['text']['trigger_words']
            targets = [
                choice['text']

                for choice in sorted(
                    line['text']['choices'],
                    key=lambda c: c['label']
                )
            ]
            
            if test_flag:
                target = line['target']
                true_answer = targets[int(answer_tempelate[target])]

            else:
                target = line['target']
                true_answer = targets[int(answer_tempelate[target])]
            f.writelines('# ::id ' + str(tid) + "\n")
            f.writelines("# ::theme " + str(trigger_words) + "\n")
            f.writelines("# ::option "+str(targets)+"\n")
            f.writelines("# ::snt "+text+"\n")
            f.writelines("# ::target " + str(target) + "\n")
            f.writelines("# ::true_answer " + true_answer + "\n")
            f.writelines("# ::save-date Fri Jan, 2020\n")
            f.write("(f / follow-02\n      :manner (i / interest-01))")
            f.writelines('\n\n')


if __name__ == '__main__':

    data_list = [
        'train_rand_split.jsonl',
        'dev_rand_split.jsonl',
        'test_rand_split.jsonl'
    ]

    test_flag = False
    for data in data_list:
        if 'test' in str(data):

            test_flag = True
        preprocess(lines=read_jsonl('./dataset/'+data), title=data, test_flag=test_flag)
    print("Execution Done..!")