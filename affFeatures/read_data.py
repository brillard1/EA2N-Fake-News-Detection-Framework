import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from features.building_features import manual_features, segmentation_text, clean_regex
from data.utils import split
import pickle
np.random.seed(0)
import sys
dataset=sys.argv[1]
output = sys.argv[2]

def prepare_input(dataset=dataset, segments_number=10, n_jobs=-1, emo_rep='frequency', return_features=True,
                  text_segments=False, clean_text=True):

    content = pd.read_csv('./data/{}/{}.csv'.format(dataset, dataset))
    content_features = []
    """Extract features, segment text, clean it."""
    if return_features:
        content_features = manual_features(n_jobs=n_jobs, path='./features', model_name=dataset,
                                           segments_number=segments_number, emo_rep=emo_rep, ids= content['id'].tolist()).transform(content['content'])
        ids = content['id'].tolist()
        content_features = content_features.flatten().reshape(len(content),240)

        data_dict = dict(zip(ids, content_features))

        with open(str(output)+".pkl", 'wb') as pickle_file:
            pickle.dump(data_dict, pickle_file)



if __name__ == '__main__':
    prepare_input(dataset=dataset, segments_number=10, n_jobs=-1, text_segments=True)