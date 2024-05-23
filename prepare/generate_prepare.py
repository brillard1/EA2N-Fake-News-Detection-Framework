import sys
import json
import ijson
import os
import shutil

TRAIN_NUM_BATCHES = int(sys.argv[2])
DEV_NUM_BATCHES = int(sys.argv[3])
TEST_NUM_BATCHES = int(sys.argv[4])


def generate_bash():

    dataset = "./amr_data/amr_2.0_politifact/ea2n"
    entity_seed = sys.argv[5]
    tokenID = 0

    with open("cmd_extract_train1.sh", 'w') as f:
        for i in range(1, 11):
            tokenID = i%5
            f.write("python3 prepare/extract_property.py --train_data %s/train.pred.txt --amr_files %s/train.pred_%d.txt --nprocessors 2 --entity_seed %s --tagme_tokenID %d > logs/log_poli_train_%d.txt &\n" %(dataset, dataset, i, entity_seed, tokenID, i))
        f.write('wait')

    with open("cmd_extract_train2.sh", 'w') as f:
        for i in range(11, 21):
            tokenID = i%5
            f.write("python3 prepare/extract_property.py --train_data %s/train.pred.txt --amr_files %s/train.pred_%d.txt --nprocessors 2 --entity_seed %s --tagme_tokenID %d > logs/log_poli_train_%d.txt &\n" %(dataset, dataset, i, entity_seed, tokenID, i))
        f.write('wait')

    with open("cmd_extract_train3.sh", 'w') as f:
        for i in range(21, 31):
            tokenID = i%5
            f.write("python3 prepare/extract_property.py --train_data %s/train.pred.txt --amr_files %s/train.pred_%d.txt --nprocessors 2 --entity_seed %s --tagme_tokenID %d > logs/log_poli_train_%d.txt &\n" %(dataset, dataset, i, entity_seed, tokenID, i))
        f.write('wait')
    #
    with open("cmd_extract_train4.sh", 'w') as f:
        for i in range(31, 41):
            tokenID = i%5
            f.write("python3 prepare/extract_property.py --train_data %s/train.pred.txt --amr_files %s/train.pred_%d.txt --nprocessors 2 --entity_seed %s --tagme_tokenID %d > logs/log_poli_train_%d.txt &\n" %(dataset, dataset, i, entity_seed, tokenID, i))
        f.write('wait')

    with open("cmd_extract_train5.sh", 'w') as f:
        for i in range(41, 51):
            tokenID = i%5
            f.write("python3 prepare/extract_property.py --train_data %s/train.pred.txt --amr_files %s/train.pred_%d.txt --nprocessors 2 --entity_seed %s --tagme_tokenID %d > logs/log_poli_train_%d.txt &\n" %(dataset, dataset, i, entity_seed, tokenID, i))
        f.write('wait')

    with open("cmd_extract_dev.sh", 'w') as f:
        for i in range(1, DEV_NUM_BATCHES+1):
            tokenID = i%5
            f.write("python3 prepare/extract_property.py --train_data %s/train.pred.txt --amr_files %s/dev.pred_%d.txt --nprocessors 2 --entity_seed %s --tagme_tokenID %d > logs/log_poli_dev_%d.txt &\n" %(dataset, dataset, i, entity_seed, tokenID, i))
        f.write('wait')

    with open("cmd_extract_test.sh", 'w') as f:
        for i in range(1, TEST_NUM_BATCHES+1):
            tokenID = i%5
            f.write("python3 prepare/extract_property.py --train_data %s/train.pred.txt --amr_files %s/test.pred_%d.txt --nprocessors 2 --entity_seed %s --tagme_tokenID %d > logs/log_poli_test_%d.txt &\n" %(dataset, dataset, i, entity_seed, tokenID, i))
        f.write('wait')

    with open("cmd_extract_exmp.sh", 'w') as f:
        for i in range(1, 2):
            f.write("python3 prepare/extract_property.py --train_data %s/train.pred.txt --amr_files %s/train.pred_%d.txt --nprocessors 2 --entity_seed %s --tagme_tokenID %d > logs/log_poli_train_%d.txt &\n" %(dataset, dataset, i, entity_seed, tokenID, i))
        f.write('wait')


# python3 prepare/extract_property.py --train_data /home/wjddn803/PycharmProjects/gct_bert/data/AMR/amr_2.0/train.pred.txt --amr_files /home/wjddn803/PycharmProjects/gct_bert/data/AMR/amr_2.0/train.pred_1.txt --nprocessors 2 --extend True --entity_seed entity_seed

def copy_files(source, destination):
    # importing shutil module

    # Copy the content of
    # source to destination

    try:
        shutil.copyfile(source, destination)
        print("File copied successfully.")

        # If source and destination are same
    except shutil.SameFileError:
        print("Source and destination represents the same file.")

        # If destination is a directory.
    except IsADirectoryError:
        print("Destination is a directory.")

        # If there is any permission issue
    except PermissionError:
        print("Permission denied.")

        # For other errors
    except:
        print("Error occurred while copying file.")

def try_parse(filename):

    # Source path
    source = filename
    # Destination path
    destination = filename[:filename.index('final.json')]+'final_original.json'
    copy_files(source, destination)
    with open(filename, 'rb+') as filehandle:
        filehandle.seek(-2, os.SEEK_END)
        filehandle.truncate()
    open(filename, 'a').write("]")


def combine():
    mode = sys.argv[5]

    PATH = './mnt/wiki_data_politifact/amr_2.0/ea2n/'

    if mode == 'dev':
        with open(PATH + "dev_pred_wiki_extended_real_final.json", 'w') as fj:
            fj.write('[')
            for i in range(1, DEV_NUM_BATCHES+1):
                print('i th batch', i)
                try_parse(PATH + "dev.pred_%d_wiki_extended_final.json" % i)
                print('done_parsing')
                with open(PATH + "dev.pred_%d_wiki_extended_final.json" % i, 'rb') as fp:
                    objects = ijson.items(fp, 'item')
                    for i, line in enumerate(objects):
                        json.dump(line, fj)
                        fj.write(' ,')


        source = PATH + 'dev_pred_wiki_extended_real_final.json'
        with open(source, 'rb+') as fj_filehandle:
            # Destination path
            destination = source[:source.index('final.json')] + 'final_original.json'
            copy_files(source, destination)
            fj_filehandle.seek(-1, os.SEEK_END)
            fj_filehandle.truncate()
        open(source, 'a').write("]")

    elif mode == 'test':
        with open(PATH + "test_pred_wiki_extended_real_final.json", 'w') as fj:
            fj.write('[')
            for i in range(1, TEST_NUM_BATCHES+1):
                print('i th batch', i)
                try_parse(PATH + "test.pred_%d_wiki_extended_final.json" % i)
                print('done_parsing')
                with open(PATH + "test.pred_%d_wiki_extended_final.json" % i, 'rb') as fp:
                    objects = ijson.items(fp, 'item')
                    for i, line in enumerate(objects):
                        json.dump(line, fj)
                        fj.write(' ,')


        source = PATH + 'test_pred_wiki_extended_real_final.json'
        with open(source, 'rb+') as fj_filehandle:
            # Destination path
            destination = source[:source.index('final.json')] + 'final_original.json'
            copy_files(source, destination)
            fj_filehandle.seek(-1, os.SEEK_END)
            fj_filehandle.truncate()
        open(source, 'a').write("]")

    else:
        with open(PATH + "train_pred_wiki_extended_real_final.json", 'w') as fj:
            fj.write('[')
            for i in range(1, TRAIN_NUM_BATCHES + 1):
                print('i th batch', i)
                try_parse(PATH + "train.pred_%d_wiki_extended_final.json" % i)
                print('done_parsing')
                with open(PATH + "train.pred_%d_wiki_extended_final.json" % i, 'rb') as fp:
                    objects = ijson.items(fp, 'item')
                    for i, line in enumerate(objects):
                        json.dump(line, fj)
                        fj.write(' ,')

        source = PATH + 'train_pred_wiki_extended_real_final.json'
        with open(source, 'rb+') as fj_filehandle:
            # Destination path
            destination = source[:source.index('final.json')] + 'final_original.json'
            copy_files(source, destination)
            fj_filehandle.seek(-1, os.SEEK_END)
            fj_filehandle.truncate()
        open(source, 'a').write("]")


if __name__ == '__main__':
    import sys
    globals()[sys.argv[1]]()
