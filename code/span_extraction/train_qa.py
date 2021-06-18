import numpy as np
import json, os, logging, pickle, argparse
from evaluate_squad import compute_f1
from simpletransformers.question_answering import QuestionAnsweringModel

def lcs(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])

    return lcs_set


def evaluate_results(text):
    partial_match_scores = []
    lcs_all = []
    impos1, impos2, impos3, impos4 = 0, 0, 0, 0
    pos1, pos2, pos3 = 0, 0, 0
    fscores, squad_fscores = [], []
    fscores_all, squad_fscores_all = [], []
    
    for i, key in enumerate(['correct_text', 'similar_text', 'incorrect_text']):
        for item in text[key]:
            if i==0:
                pos1 += 1
                fscores.append(1)
                squad_fscores.append(1)
                    
            elif i==1:  
                z = text[key][item]
                if z['predicted'] != '':
                    longest_match = list(lcs(z['truth'], z['predicted']))[0]
                    lcs_all.append(longest_match)
                    partial_match_scores.append(round(len(longest_match.split())/len(z['truth'].split()), 4))
                    pos2 += 1
                    r = len(longest_match.split())/len(z['truth'].split())
                    p = len(longest_match.split())/len(z['predicted'].split())
                    f = 2*p*r/(p+r)
                    fscores.append(f)
                    squad_fscores.append(compute_f1(z['truth'], z['predicted']))
                else:
                    pos3 += 1
                    impos4 += 1
                    fscores.append(0)
                    squad_fscores.append(0)                                
                    
            elif i==2:
                if z['predicted'] == '':
                    impos4 += 1
                pos3 += 1
                fscores.append(0)
                squad_fscores.append(0)
                    
    total_pos = pos1 + pos2 + pos3
    
    p1 = 'Postive Samples:'
    p2 = 'Exact Match: {}/{} = {}%'.format(pos1, total_pos, round(100*pos1/total_pos, 2))
    p3 = 'Partial Match: {}/{} = {}%'.format(pos2, total_pos, round(100*pos2/total_pos, 2))
    p4 = 'No Match: {}/{} = {}%'.format(pos3, total_pos, round(100*pos3/total_pos, 2))
    p5 = 'F1 Score = {}%'.format(round(100*np.mean(fscores), 2))
    p6 = 'SQuAD F1 Score = {}%'.format(round(100*np.mean(squad_fscores), 2))
    
    p = '\n'.join([p1, p2, p3, p4, p5, p6])
    return p


if __name__ == '__main__':

    global args
    parser = argparse.ArgumentParser()
     
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='Initial learning rate') 
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=12, metavar='E', help='number of epochs')
    parser.add_argument('--model', default='rob', help='which model rob| robsq | span')
    parser.add_argument('--cuda', type=int, default=0, metavar='C', help='cuda device')
    parser.add_argument('--fold', type=int, default=0, metavar='F', help='which fold')
    args = parser.parse_args()

    print(args)
    
    model_family = {'rob': 'roberta', 'robsq': 'roberta', 'span': 'bert'}
    model_id = {'rob': 'roberta-base', 'robsq': 'roberta-base-sqaud', 'span': 'spanbert-squad'}
    model_exact_id = {'rob': 'roberta-base', 'robsq': 'deepset/roberta-base-squad2', 
                      'span': 'mrm8488/spanbert-finetuned-squadv2'}
    
    
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    model = args.model
    cuda = args.cuda
    fold = str(args.fold)
    
    max_q_length, max_c_length, max_a_length = 50, 400, 50
    
    x_train = json.load(open('data/fold' + fold + '_train.json'))
    x_test  = json.load(open('data/fold' + fold + '_test.json'))
    
    save_dir    = 'outputs/' + model_id[model] + '/fold' + fold + '/'
    result_file = 'outputs/' + model_id[model] + '/fold' + fold + '_results.txt'
    prediction_file   = 'outputs/' + model_id[model] + '/fold' + fold + '_test_predictions.pkl' 
    
    num_steps = int(3500/batch_size)
    
    train_args = {
        'fp16': False,
        'overwrite_output_dir': True, 
        'doc_stride': 512, 
        'max_query_length': max_q_length, 
        'max_answer_length': max_a_length,
        "max_seq_length": max_c_length,
        'n_best_size': 20,
        'null_score_diff_threshold': 0.0,
        'learning_rate': lr,
        'sliding_window': False,
        'output_dir': save_dir,
        'best_model_dir': save_dir + 'best_model/',
        'evaluate_during_training': True,
        'evaluate_during_training_steps': num_steps,
        'save_eval_checkpoints': False,
        'save_model_every_epoch': False,
        'save_steps': 500000,
        'train_batch_size': batch_size,
        'num_train_epochs': epochs
    }
    
    qa_model = QuestionAnsweringModel(model_family[model], model_exact_id[model], args=train_args, cuda_device=cuda)
    qa_model.train_model(x_train, eval_data=x_test)
    qa_model = QuestionAnsweringModel(model_family[model], save_dir + 'best_model/', args=train_args, cuda_device=cuda)
    
    result, text = qa_model.eval_model(x_test)
    r = evaluate_results(text)
    print('Fold ' + fold + ' performance.')
    print (r)
    
    rf = open('results/qa_model_' + model_id[model] + '.txt', 'a')
    
    rf.write(str(args) + '\n\n')
    rf.write(r + '\n' + '-'*40 + '\n')    
    rf.close()
    
    rf = open(result_file, 'a')
    rf.write(str(args) + '\n\n')
    rf.write(r + '\n' + '-'*40 + '\n')    
    rf.close()
    
    pickle.dump(text, open(prediction_file, 'wb'))
    
