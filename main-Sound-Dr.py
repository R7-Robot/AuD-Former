import torch
import argparse
from src.utils.utils import *
from torch.utils.data import DataLoader
from src import train
from src import test
 

parser = argparse.ArgumentParser(description='Audformer Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='Audformer',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--dataset', type=str, default='Audformer',
                    help='dataset to use (default: Audformer)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.05,
                    help='attention dropout')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.1,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=3,
                    help='number of heads for the transformer network (default: 3)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 1024)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer to use (default: SGD)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=5,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='aud',
                    help='name of the trial (default: "aud")')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
args = parser.parse_args()

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())

use_cuda = False

output_dim_dict = {
    'Audformer': 1
}

criterion_dict = {
    'Audformer': 'CrossEntropyLoss'
}

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

####################################################################
#
# Load the dataset 
#
####################################################################

print("Start loading the data....")

train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')
   
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,generator=torch.Generator(device = 'cuda'))
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True,generator=torch.Generator(device = 'cuda'))
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True,generator=torch.Generator(device = 'cuda'))

print('Finish loading the data....')

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.orig_d_m1, hyp_params.orig_d_m2, hyp_params.orig_d_m3,hyp_params.orig_d_m4,hyp_params.orig_d_m5, hyp_params.orig_d_m6,hyp_params.orig_d_m7 = train_data.get_dim()
hyp_params.orig_d_m8, hyp_params.orig_d_m9, hyp_params.orig_d_m10,hyp_params.orig_d_m11,hyp_params.orig_d_m12, hyp_params.orig_d_m13,hyp_params.orig_d_m14 = train_data.get_dim1()
hyp_params.orig_d_m15, hyp_params.orig_d_m16, hyp_params.orig_d_m17,hyp_params.orig_d_m18,hyp_params.orig_d_m19, hyp_params.orig_d_m20,hyp_params.orig_d_m21 = train_data.get_dim2()
hyp_params.m1_len, hyp_params.m2_len, hyp_params.m3_len, hyp_params.m4_len, hyp_params.m5_len, hyp_params.m6_len, hyp_params.m7_len = train_data.get_seq_len()
hyp_params.m8_len, hyp_params.m9_len, hyp_params.m10_len, hyp_params.m11_len, hyp_params.m12_len, hyp_params.m13_len, hyp_params.m14_len = train_data.get_seq_len1()
hyp_params.m15_len, hyp_params.m16_len, hyp_params.m17_len, hyp_params.m18_len, hyp_params.m19_len, hyp_params.m20_len, hyp_params.m21_len = train_data.get_seq_len2()
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')


if __name__ == '__main__':
    if args.eval:
        test = test.eval(hyp_params, test_loader)
    else:
        test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)

