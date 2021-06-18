from tqdm import tqdm
import numpy as np
import argparse, time, pickle
import torch
import torch.optim as optim
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from dataloader import NLILoader
from model import TransformerNLIModel

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report

def configure_optimizers(model, weight_decay, learning_rate, adam_epsilon, warmup_steps, t_total):
    "Prepare optimizer"
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    
    return optimizer, scheduler


def configure_dataloaders(mode, fold, batch_size):
    "Prepare dataloaders"
    
    train_file = 'data/fold' + str(fold) + '_w_neg_train_lemma.tsv'
    test_file = 'data/fold' + str(fold) + '_w_neg_test_lemma.tsv'
    
    print ('Train on:', train_file)
    print ('Test on:', test_file)
    
    train_loader = NLILoader(
        train_file, 
        mode,
        batch_size,
        shuffle=True
    )
    
    test_loader = NLILoader(
        test_file, 
        mode,
        batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, optimizer=None, scheduler=None, train=False):
    losses, preds, labels = [], [], []
    assert not train or optimizer!=None
    
    if train:
        model.train()
    else:
        model.eval()
    
    for step, batch in enumerate(tqdm(dataloader, leave=False)):
        
        text, label = batch
        label = torch.tensor(label).cuda()
        
        if train:
            optimizer.zero_grad()
           
        # obtain log probabilities
        log_prob = model(text)
        pred = torch.argmax(log_prob, 1)
        
        # compute loss
        loss = loss_function(log_prob, label)

        # accumulate results
        preds.append(pred.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())
        losses.append(loss.item())
        
        if train:
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            loss.backward()
            optimizer.step()
            scheduler.step()

    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), float('nan'), [], []

    avg_loss = round(np.mean(losses), 4)
    accuracy = round(accuracy_score(labels, preds)*100, 2)
    fscore1 = round(f1_score(labels, preds, average='macro')*100, 2)
    fscore2 = round(f1_score(labels, preds, average='weighted')*100, 2)
        
    return avg_loss, accuracy, fscore1, fscore2, labels, preds

if __name__ == '__main__':

    global args
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--warmup-steps", default=-1, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")    
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='Initial learning rate')
    parser.add_argument('--weight_decay', default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--adam-epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")  
    parser.add_argument('--batch-size', type=int, default=8, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=10, metavar='E', help='number of epochs')
    parser.add_argument('--mode', default='0', help='which model 0: roberta-mnli | 1: roberta-mnli-anli-fever | 2: roberta')
    parser.add_argument('--fold', type=int, default=1, help='which fold 1|2|3|4|5')
    args = parser.parse_args()

    print(args)

    global dataset
    batch_size = args.batch_size
    n_epochs = args.epochs
    mode = args.mode
    fold = args.fold
    
    model = TransformerNLIModel(mode).cuda()
    loss_function = torch.nn.NLLLoss()
        
    train_loader, test_loader = configure_dataloaders(mode, fold, batch_size)
    
    estimated_training_size = 10000
    t_total = estimated_training_size * 15 // batch_size
    
    optimizer, scheduler = configure_optimizers(model, args.weight_decay, args.lr, args.adam_epsilon, args.warmup_steps, t_total)
    
    lf = open('logs/dialogue_nli_mode_' + mode + '_fold_' + str(fold) + '.txt', 'a')
    rf = open('results/dialogue_nli_mode_' + mode + '_fold_' + str(fold) + '.txt', 'a')
    
    lf.write(str(args)+'\n\n')  
    test_losses, test_fscores1, test_fscores2, test_accuracy = [], [], [], []
    best_score, best_label, best_pred = 0, None, None

    for e in range(n_epochs):
        start_time = time.time()
        
        train_loss, train_acc, train_fscore1, train_fscore2, _, _ = train_or_eval_model(model, loss_function, train_loader, optimizer, scheduler, True)
        
        test_loss, test_acc, test_fscore1, test_fscore2, test_label, test_pred = train_or_eval_model(model, loss_function, test_loader)
        
        test_losses.append(test_loss)
        test_fscores1.append(test_fscore1)
        test_fscores2.append(test_fscore2)
        test_accuracy.append(test_acc)
        
        if best_score < test_fscore1:
            best_score, best_label, best_pred = test_fscore1, test_label, test_pred
            
        x = 'Epoch {} train loss {} acc {} macro-f1 {} w-f1 {}; test loss {} acc {} macro-f1 {} w-f1 {}; time {}'.\
                    format(e+1, train_loss, train_acc, train_fscore1, train_fscore2, \
                           test_loss, test_acc, test_fscore1, test_fscore2, round(time.time()-start_time, 2))
        lf.write(x + '\n')  
        print (x)
        
    test_fscores1 = np.array(test_fscores1).transpose()
    test_fscores2 = np.array(test_fscores2).transpose()
    test_accuracy = np.array(test_accuracy).transpose()
        
    print('Fold ' + str(fold) + ' performance.')
    score1 = test_fscores1[np.argmin(test_losses)]
    score2 = test_fscores1[np.argmax(test_fscores1)]
    score3 = test_fscores2[np.argmin(test_losses)]
    score4 = test_fscores2[np.argmax(test_fscores2)]
    score5 = test_accuracy[np.argmin(test_losses)]
    score6 = test_accuracy[np.argmax(test_accuracy)]
    
    if mode  == '0':
        entail_class = [2]
    else:
        entail_class = [0]
        
    score7 = round(precision_recall_fscore_support(best_label, best_pred, labels=entail_class)[0][0]*100, 2)
    score8 = round(precision_recall_fscore_support(best_label, best_pred, labels=entail_class)[1][0]*100, 2)
    
    # pickle.dump([best_label, best_pred], open('predictions/csk_nli_' + '_mode_' + mode + '_fold_' + str(fold) + '.pkl', 'wb'))
    
    print('Entailment@Best Valid F1: Precision {}; Recall {}'.format(score7, score8))
    print('Macro F1@Best Valid Loss: {}; Macro F1@Best Valid F1: {}'.format(score1, score2))
    print('Weighted F1@Best Valid Loss: {}; Macro F1@Best Valid F1: {}'.format(score3, score4))
    print('Acc@Best Valid Loss: {}; Acc@Best Valid Acc: {}'.format(score5, score6))
    
    scores = [score1, score2, score3, score4, score5, score6, score7, score8]
    scores = [str(item) for item in scores]
        
    print(classification_report(best_label, best_pred, digits=4))
    print(confusion_matrix(best_label, best_pred))
        
    lf.write(str(classification_report(best_label, best_pred, digits=4)) + '\n')
    lf.write(str(confusion_matrix(best_label, best_pred)) + '\n')
    lf.write('\t'.join(scores) + '\n')
    lf.write('-'*50 + '\n\n')
    lf.close()
    
    rf.write('\t'.join(scores) + '\t' + str(args) + '\n')
    rf.close()
    