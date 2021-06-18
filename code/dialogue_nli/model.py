import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification  
    
class TransformerNLIModel(nn.Module):
    def __init__(
        self,
        mode
    ):
        super().__init__()
        
        self.mode = mode
        
        if mode == '0':
            # {0: "contradiction", 1: "neutral", 2: "entailment"}
            model_name = "roberta-large-mnli"
            print ('RoBERTa Large MNLI Model.')
        
        elif mode == '1':
            # {0: "entailment", 1: "neutral", 2: "contradiction"}
            model_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
            print ('RoBERTa Large SNLI-MNLI-ANLI-FEVER Model.')
            
        elif mode == '2':
            #  {"entailment": 0, "contradiction": 1}
            model_name = "roberta-large"
            print ('RoBERTa Large Model.')
            self.dense = nn.Linear(1024, 2).cuda()
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()
        
    def forward(
        self, 
        text
    ):  
        batch = self.tokenizer(text, padding=True, return_tensors="pt")
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        output = self.model(input_ids[:, :512], attention_mask[:, :512], output_hidden_states=True)
        if self.mode == 2:
            comb = output.hidden_states[-1][:,0,:]
            comb2 = self.dense(comb)
            log_prob = torch.nn.functional.log_softmax(comb2, dim=1)
        else:
            log_prob = torch.nn.functional.log_softmax(output[0], dim=1)
        
        return log_prob
    