import os
import sys

if __name__ == "__main__":

    # Select Dataset
    data = input("Select the dataset name:\n1. Politifact\n2. Gossipcop\nOption: ")
    if data == "1":
        dataset = 'politifact'
    elif data == "2":
        dataset = 'gossipcop'

    # Select Model
    mod = input("Select the model name:\n1. Bert\n2. Roberta\n3. Electra\n4. XLNet\nOption: ")
    if mod == "1":
        model = 'EA2N_bert_dual'
    elif mod == "2":
        model = 'EA2N_roberta_dual'
    elif mod == "3":
        model = 'EA2N_electra_dual'
    elif mod == "4":
        model = 'EA2N_xlnet_dual'

    # Select AMR or No AMR
    app = input("Select approach 1:\n1. AMR\n2. No AMR\nOption: ")
    if app == "1":
        amr = 'True'
    elif app == "2":
        amr = 'False'

    affective_features = 'True'

    # Selecting gpu
    curr_gpu = 0
    mode = 'train'
    if (len(sys.argv) == 1):
      curr_gpu = 0
    elif (len(sys.argv) == 3):
      curr_gpu = sys.argv[1]
      mode = sys.argv[2]
    else:
        if (sys.argv[1] == 'train'):
            mode = 'train'
        elif (sys.argv[1] == 'eval'):
            mode = 'eval'

    # Execute the sh command
    os.system(f"CUDA_VISIBLE_DEVICES=0,1 python3 -u main.py --mode {mode} --encoder_type {model} --dataset {dataset} --amr {amr} --affFeatures {affective_features} --gpus {curr_gpu}")
