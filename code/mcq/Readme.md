## Data Gerneration
To generate the data for adversarial training, please check the readme in the [`data_generation`](data_generation/Readme.md) folder. 

We have provided data for 35 iterations of adversarial training (n_iter=35), for both 5 folds (0,1,2,3,4), so that you can use directly.

## Setup
The code is tested on pytorch=1.8 and python=3.8 First install all the requirements from requirements.txt

## Run code


### Bert Base with context
```sh
python main.py   --train_file dataset/train_iter{n_iter}_fold{n_fold}.csv   --validation_file dataset/test_iter{n_iter}_fold{n_fold}.csv --model_name_or_path bert-base-uncased  --do_train    --do_eval    --learning_rate 5e-5   --num_train_epochs 3   --max_seq_length 350   --output_dir outputs/bert/fold{n_fold}   --per_gpu_eval_batch_size=8   --per_gpu_train_batch_size=8   --gradient_accumulation_steps 2   --overwrite_output;
```


### Roberta Base with context:
```sh
python main.py   --train_file dataset/train_iter{n_iter}_fold{n_fold}.csv   --validation_file dataset/test_iter{n_iter}_fold{n_fold}.csv --model_name_or_path roberta-base  --do_train    --do_eval    --learning_rate 1e-5   --num_train_epochs 10   --max_seq_length 350   --output_dir outputs/roberta/fold{n_fold}   --per_gpu_eval_batch_size=8   --per_gpu_train_batch_size=8   --gradient_accumulation_steps 5   --overwrite_output;
```

### Bert Base without context
```sh
python main.py   --train_file dataset/train_iter{n_iter}_fold{n_fold}_Q.csv   --validation_file dataset/test_iter{n_iter}_fold{n_fold}_Q.csv --model_name_or_path bert-base-uncased  --do_train    --do_eval    --learning_rate 5e-5   --num_train_epochs 3   --max_seq_length 350   --output_dir outputs/bert/fold{n_fold}_Q   --per_gpu_eval_batch_size=8   --per_gpu_train_batch_size=8   --gradient_accumulation_steps 2   --overwrite_output;
```

### Roberta without context
```sh
	python main.py   --train_file dataset/train_iter{n_iter}_fold{n_fold}_Q.csv   --validation_file dataset/test_iter{n_iter}_fold{n_fold}_Q.csv --model_name_or_path roberta-base  --do_train    --do_eval    --learning_rate 5e-5   --num_train_epochs 3   --max_seq_length 350   --output_dir outputs/roberta/fold{n_fold}_Q   --per_gpu_eval_batch_size=8   --per_gpu_train_batch_size=8   --gradient_accumulation_steps 2   --overwrite_output;
```

### Roberta pretraiend on squad with context:
```sh
python main.py   --train_file dataset/train_iter{n_iter}_fold{n_fold}.csv   --validation_file dataset/test_iter{n_iter}_fold{n_fold}.csv --model_name_or_path deepset/roberta-base-squad2  --do_train    --do_eval    --learning_rate 1e-5   --num_train_epochs 10   --max_seq_length 350   --output_dir outputs/roberta/fold{n_fold}-squad   --per_gpu_eval_batch_size=8   --per_gpu_train_batch_size=8   --gradient_accumulation_steps 5   --overwrite_output;
```
