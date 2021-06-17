
For each iteration, run the following code
python main.py   --train_file dataset/train_iter10.csv   --validation_file dataset/val_iter10.csv --model_name_or_path roberta-base --do_train    --do_eval    --learning_rate 5e-5   --num_train_epochs 3   --max_seq_length 350   --output_dir race_out/bert_base_uncased   --per_gpu_eval_batch_size=8   --per_gpu_train_batch_size=8   --gradient_accumulation_steps 2   --overwrite_output;

## The path in the jupyter notebook may need to be changed accordingly
1. Train a spanBert model on `original_data/relations_only/train_and_test.json` by running script 
```sh
python train.py --cuda 2 --fold 1 --model span --epochs 15
```
> Note: The trained model is stored in `adversarial_outputs` folder
After the model is trained, we can generate the negative pool by using the trained model on `original_dat/relations_only/adversarial_train_and_test.json` to get `predictions_test.json` which contains the negative pool, by running script
```sh
python predict.py --cude 2 --fold 1 --model span --epochs 0
```
3. Run section1 in `adversarial_training.ipynb` to generate `negative_options.json`
4. Run section2 in `adversarial_training.ipynb` to generate `processed_data/train_iter0.csv` and `processed_data/val_iter0.csv` to generated the initial training files
6. Mv data to `../dataset` folder and train a model
7. Run Section3 in jupyter notebook
8. repeat 6 and 7 for the desired iterations
9. Run Section5 to split the data for 5 folds.
