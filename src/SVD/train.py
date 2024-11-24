import torch
from torch import nn
import sys
from src import models
from src.utils.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.utils.eval_metrics import *


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params.model+'Model')(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr,momentum = 1,nesterov=True)
    criterion = getattr(nn, hyp_params.criterion)()
    #criterion = BCEFocalLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    

    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        mae_train2 = 0
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
            sample_ind, m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,m31,m32,m33,m34,m35 = batch_X
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            
            model.zero_grad()
            if hyp_params.use_cuda:
                #print('111111')
                with torch.cuda.device(0):
                    m1,m2,m3,m4,m5,m6,m7,eval_attr = m1.cuda(),m2.cuda(),m3.cuda(),m4.cuda(),m5.cuda(),m6.cuda(),m7.cuda(),eval_attr.cuda()
                    m8,m9,m10,m11,m12,m13,m14 = m8.cuda(),m9.cuda(),m10.cuda(),m11.cuda(),m12.cuda(),m13.cuda(),m14.cuda()
                    m15,m16,m17,m18,m19,m20,m21 = m15.cuda(),m16.cuda(),m17.cuda(),m18.cuda(),m19.cuda(),m20.cuda(),m21.cuda()
                    m22,m23,m24,m25,m26,m27,m28 = m22.cuda(),m23.cuda(),m24.cuda(),m25.cuda(),m26.cuda(),m27.cuda(),m28.cuda()
                    m29,m30,m31,m32,m33,m34,m35 = m29.cuda(),m30.cuda(),m31.cuda(),m32.cuda(),m33.cuda(),m34.cuda(),m35.cuda()
            
            batch_size = m1.size(0)
            batch_chunk = hyp_params.batch_chunk

            combined_loss = 0
            net = nn.DataParallel(model) if batch_size > 10 else model
            
            preds, hiddens = net(m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,m31,m32,m33,m34,m35)
            raw_loss = criterion(preds, eval_attr)
            combined_loss = raw_loss 
            combined_loss.backward()
            mae_train1 = mae1(preds,eval_attr)
            mae_train2 += mae_train1
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f} | memory_used {:5.4f} MB'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss,memory_used))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
                
        mae_train = mae_train2 / num_batches
        print('mae_train:',mae_train)
        return epoch_loss / hyp_params.n_train, mae_train

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
    
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                sample_ind,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,m31,m32,m33,m34,m35 = batch_X
                eval_attr = batch_Y.squeeze(dim=-1) # if num of labels is 1
            
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        m1,m2,m3,m4,m5,m6,m7,eval_attr = m1.cuda(),m2.cuda(),m3.cuda(),m4.cuda(),m5.cuda(),m6.cuda(),m7.cuda(),eval_attr.cuda()
                        m8,m9,m10,m11,m12,m13,m14 = m8.cuda(),m9.cuda(),m10.cuda(),m11.cuda(),m12.cuda(),m13.cuda(),m14.cuda()    
                        m15,m16,m17,m18,m19,m20,m21 = m15.cuda(),m16.cuda(),m17.cuda(),m18.cuda(),m19.cuda(),m20.cuda(),m21.cuda()
                        m22,m23,m24,m25,m26,m27,m28 = m22.cuda(),m23.cuda(),m24.cuda(),m25.cuda(),m26.cuda(),m27.cuda(),m28.cuda()  
                        m29,m30,m31,m32,m33,m34,m35 = m29.cuda(),m30.cuda(),m31.cuda(),m32.cuda(),m33.cuda(),m34.cuda(),m35.cuda()
                batch_size = m1.size(0)
                net = nn.DataParallel(model) if batch_size > 10 else model
                preds, _ = net(m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,m31,m32,m33,m34,m35)
                total_loss += criterion(preds, eval_attr).item() * batch_size

                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)
                
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths
    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        _,mae_train = train(model, optimizer, criterion)
        val_loss, _, _ = evaluate(model,criterion, test=False)
        test_loss, _, _ = evaluate(model,criterion, test=True)
        end = time.time()
        duration = end-start
        scheduler.step(val_loss)    # Decay learning rate by validation loss
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f} | memory_used{:5.4f} MB'.format(epoch, duration, val_loss, test_loss,memory_used))
        print("-"*50)
        if val_loss < best_valid:
            print(f"Saved model at output/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths,_ = evaluate(model, criterion, test=True)
    eval_hus(results, truths, True)


