### Execution

To train and evaluate models in the Span Extraction task:

```bash
python train_qa.py --cuda 0 --fold [1|2|3|4|5] --model [span|rob]
```

`--model span` will fine-tune a SQuAD SpanBERT model and `--model rob` will fine-tune a RoBERTa model.

Please check [`train_qa.py`](train_qa.py) for more details.

The `simpletransformers` library is used from [here](https://github.com/ThilinaRajapakse/simpletransformers). The `evaluate_squad.py` is the evaluation script from the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/).