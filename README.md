# EA2N: Evidence-based AMR Attention Network for Fake News Detection
Code for the above paper

## Environment Setup

Install the required dependencies

``` 
pip install -r requirements.txt
```

OR

Setup Conda Environment

```
conda env create -f environment.yml
```

## Data Preparation
In this module we'll download the dataset and generate AMR linked with WikiData (referred as WikiAMR)
### Data Preprocessing
- Download FakeNewsNet Dataset by following the steps mentioned [here](https://github.com/KaiDMML/FakeNewsNet) and place it in 'dataset' folder

- Download the WikiData5m from [here](https://pykeen.readthedocs.io/en/stable/api/pykeen.datasets.Wikidata5M.html) and place it in 'wikinet' folder

- Download Artifacts for glove embeddings and other utilities
    ```
    ./scripts/download_artifacts.sh
    ```

- cd in 'dataset' and run
    ```
    python preprocess.py
    ```
- Split data into train, dev and test using
    ```
    python scripts/parser_fnn.py {dataset_folder} {politifact/gossipcop}
    ```
- Prepare data for train, dev and test
    ```
    cd ..
    ./scripts/prepare_data.sh -v 2 -p [project_path]
    ```
- We use Stanford CoreNLP (version 3.9.2) for tokenizing.

    - Start a CoreNLP server.

    - Annotate news article sentences
        ```
        ./scripts/annotate_features.sh amr_data/amr_2.0/csqa
        ```
- Preprocess this data
    ```
    ./scripts/preprocess_2.0.sh
    ```
    this step requires [stog](https://github.com/sheng-z/stog) folder which is included in our codebase
- Generate AMR for news articles using:
    ```
    cd prepare/
    python amr_generation.py {fin.txt} {fout.txt}
    ```
    fin is train, dev or test and fout is the output file\
    Or use the stog model from [here](https://github.com/sheng-z/stog) to generate it
- Affective Features
    - Create Affective Dataloader
        ```
        cd affective_features/
        python aff_dataloader.py {dataset} {data_file}
        ```
    - Generate affFeatures
        ```
        python read_data.py {dataset} {aff_embeddings}
        ```
### WikiAMR Integration Module
#### Dividing our data into bathches to attach wikidata
---
It will takes some time as we write all the paths of the WikiAMR graph. This will need enoguh space to save the data (mnt folder would be fine choice)
```
cd prepare
python generate_batch.py 50 10 10 <Dataset Name># train/dev/test
cd ..
bash wikidata_integration.sh
```
#### Creating Vocab
---
The vocab will be created from AMR entities and the linked Wikidata paths
```
sh ./prepare/prepare.sh
```
#### Combine the divided files
---
```
python generate_prepare.py combine 50 10 10 # train/dev/test
```

## Train

Train the model by running `run.py` with mode `train`. This will prompt custom pipeline which the user wants to train the model on
```
python run.py {gpu} train
```
## Evaluate

Evaluate the model by running `run.py` with mode `eval`. This will prompt custom pipeline which the user wants to evaluate the model on
```
python run.py {gpu} eval
```
