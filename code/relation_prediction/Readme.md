## Data format conversion
The code to convert data from original data format in CIDER to this task is in the `ipynb/` folder.

## Setup
The code is tested on pytorch=1.8 and python=3.8
First install all the requirements from requirements.txt


## Run code

The following code automatically trains and tests on different folds

### Bert for fold_number from range 1 to 5
```sh
python main.py   --model_name_or_path bert-base-uncased   --train_file datasets/csk_rp/fold{fold_number}_train_lemma.csv --validation_file datasets/csk_rp/fold{fold_number}_test_lemma.csv  --do_train   --do_eval   --max_seq_length 350   --per_device_train_batch_size 32   --num_train_epochs 40   --output_dir ./outputs/bert-fold{fold_number}
```
### Robert for fold_number from range 1 to 5
```sh
python main.py   --model_name_or_path roberta-base   --train_file datasets/csk_rp/fold{fold_number}_train_lemma.csv --validation_file datasets/csk_rp/fold{fold_number}_test_lemma.csv  --do_train   --do_eval   --max_seq_length 350   --per_device_train_batch_size 32   --num_train_epochs 40   --output_dir ./outputs/roberta-fold{fold_number}
```
### Roberta without context for fold_number from range 1 to 5
```sh
python main.py   --model_name_or_path roberta-base   --train_file datasets/csk_rp_noContext/fold{fold_number}_train_lemma.csv --validation_file datasets/csk_rp_noContext/fold{fold_number}_test_lemma.csv  --do_train   --do_eval   --max_seq_length 50   --per_device_train_batch_size 32   --num_train_epochs 20   --output_dir ./outputs/roberta-noContext-fold{fold_number} 
```
