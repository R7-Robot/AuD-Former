from src.utils.eval_metrics import *
from src.utils.utils import *
from torch.utils.data import DataLoader
from torch import nn
def eval(hyp_params, test_loader):
    model = load_model(hyp_params, name=hyp_params.name)
    model.eval()
    loader = test_loader 
    total_loss = 0.0
    criterion = getattr(nn, hyp_params.criterion)()
    results = []
    truths = []

    with torch.no_grad():
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
            sample_ind, m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,m21,m22,m23,m24,m25,m26,m27,m28 = batch_X
            eval_attr = batch_Y.squeeze(dim=-1) # if num of labels is 1
            
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    m1,m2,m3,m4,m5,m6,m7,eval_attr = m1.cuda(),m2.cuda(),m3.cuda(),m4.cuda(),m5.cuda(),m6.cuda(),m7.cuda(),eval_attr.cuda()
                    m8,m9,m10,m11,m12,m13,m14 = m8.cuda(),m9.cuda(),m10.cuda(),m11.cuda(),m12.cuda(),m13.cuda(),m14.cuda() 
                    m15,m16,m17,m18,m19,m20,m21 = m15.cuda(),m16.cuda(),m17.cuda(),m18.cuda(),m19.cuda(),m20.cuda(),m21.cuda()
                    m22,m23,m24,m25,m26,m27,m28 = m22.cuda(),m23.cuda(),m24.cuda(),m25.cuda(),m26.cuda(),m27.cuda(),m28.cuda()
                        
            batch_size = m1.size(0)
            net = nn.DataParallel(model) if batch_size > 10 else model
            preds, _ = net(m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,m21,m22,m23,m24,m25,m26,m27,m28)
            total_loss += criterion(preds, eval_attr).item() * batch_size


            results.append(preds)
            truths.append(eval_attr)
                
    avg_loss = total_loss / hyp_params.n_test 

    results = torch.cat(results)
    truths = torch.cat(truths)


    eval_hus(results, truths, True)
    return avg_loss, results, truths




