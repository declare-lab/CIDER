# CIDER: Commonsense Inference for Dialogue Explanation and Reasoning

This repository contains the dataset and the pytorch implementations of the models from the paper CIDER: Commonsense Inference for Dialogue Explanation and Reasoning. CIDER has been accepted to appear at SIGDIAL 2021. The preprint version can be found [here](https://arxiv.org/abs/2106.00510).


![Alt text](cider.jpg?raw=true "Annotations in CIDER")

Commonsense inference to understand and explain human language is a fundamental research problem in natural language processing. Explaining human conversations poses a great challenge as it requires contextual understanding, planning, inference, and several aspects of reasoning including causal, temporal, and commonsense reasoning. In this work, we introduce CIDER -- a manually curated dataset that contains dyadic dialogue explanations in the form of implicit and explicit knowledge triplets inferred using contextual commonsense inference. Extracting such rich explanations from conversations can be conducive to improving several downstream applications. The annotated triplets are categorized by the type of commonsense knowledge present (e.g., causal, conditional, temporal). We set up three different tasks conditioned on the annotated dataset: Dialogue-level Natural Language Inference, Span Extraction, and Multi-choice Span Selection. Baseline results obtained with transformer-based models reveal that the tasks are difficult, paving the way for promising future research. 


## Dataset

The original annotated dataset can be found in the json files in the `data` folder.

### Data Format

Each instance in the JSON file is a dictionary of the following items:   

| Key                                | Value                                                                        | 
| ---------------------------------- |:----------------------------------------------------------------------------:| 
| `id`                               | Id of the dialogue in DailyDialog, DREAM, or MuTual.                         |
| `utterances`                       | Utterances of the dialogue spoken by speaker A or B.                         | 
| `triplets`                         | List of annotated triplets.                                                  | 
|     `head`                         | Head span of the triplet.                                                    | 
|     `relation`                     | Relation of the triplet.                                                     | 
|     `tail`                         | Tail span of the triplet.                                                    |


Example format in JSON:

```json
{
        "id": "daily-dialogue-1063",
        "utterances": "A: Gordon , you're ever so late .    B: Yes , I am sorry . I missed the bus .    A: But there's a bus every ten minutes , and you are over 1 hour late .    B: Well , I missed several buses .    A: How on earth can you miss several buses ?    B: I , ah ... , I got have late .    A: Oh , come on , Gordon , it's the afternoon now . Why were you late really ?    B: Well , I ... I lost my wallet , and I ...    A: Have you got it now ?    B: Yes , I found it again .    A: When ?    B: This morning . I mean ...    A: This tardiness causes embarrassment every time . ",
        "triplets": [
            {
                "head": "missed the bus",
                "relation": "Causes",
                "tail": "late"
            },
            {
                "head": "lost my wallet",
                "relation": "Causes",
                "tail": "late"
            },
            {
                "head": "bus",
                "relation": "HappensIn",
                "tail": "every ten minutes"
            },
            {
                "head": "missed several buses",
                "relation": "Causes",
                "tail": "over 1 hour late"
            },
            {
                "head": "lost my wallet",
                "relation": "Before",
                "tail": "found it again"
            },
            {
                "head": "tardiness",
                "relation": "causes",
                "tail": "embarrassment"
            }
        ]
    }
```


## Citation

```bash
CIDER: Commonsense Inference for Dialogue Explanation and Reasoning. Deepanway Ghosal and Pengfei Hong and Siqi Shen and Navonil Majumder and Rada Mihalcea and Soujanya Poria. SIGDIAL 2021.
```
