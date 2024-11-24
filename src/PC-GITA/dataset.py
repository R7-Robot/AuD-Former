import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    


class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='Audformer', split_type='train'):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data+'.pkl'  )
        dataset = pickle.load(open(dataset_path, 'rb'))

        # These are torch tensors
        self.m1 = torch.tensor(dataset[split_type]['DDK_mfcc'].astype(np.float32)).cpu().detach()
        self.m2 = torch.tensor(dataset[split_type]['DDK_rms'].astype(np.float32)).cpu().detach()
        self.m3 = torch.tensor(dataset[split_type]['DDK_zcr'].astype(np.float32)).cpu().detach()
        self.m4 = torch.tensor(dataset[split_type]['DDK_cenrtoid'].astype(np.float32)).cpu().detach()
        self.m5 = torch.tensor(dataset[split_type]['DDK_log_mel'].astype(np.float32)).cpu().detach()
        self.m6 = torch.tensor(dataset[split_type]['DDK_gfcc'].astype(np.float32)).cpu().detach()
        self.m7 = torch.tensor(dataset[split_type]['DDK_cqcc'].astype(np.float32)).cpu().detach()
        self.m8 = torch.tensor(dataset[split_type]['sentence_mfcc'].astype(np.float32)).cpu().detach()
        self.m9 = torch.tensor(dataset[split_type]['sentence_rms'].astype(np.float32)).cpu().detach()
        self.m10 = torch.tensor(dataset[split_type]['sentence_zcr'].astype(np.float32)).cpu().detach()
        self.m11 = torch.tensor(dataset[split_type]['sentence_cenrtoid'].astype(np.float32)).cpu().detach()
        self.m12 = torch.tensor(dataset[split_type]['sentence_log_mel'].astype(np.float32)).cpu().detach()
        self.m13 = torch.tensor(dataset[split_type]['sentence_gfcc'].astype(np.float32)).cpu().detach()
        self.m14 = torch.tensor(dataset[split_type]['sentence_cqcc'].astype(np.float32)).cpu().detach()
        self.m15 = torch.tensor(dataset[split_type]['vowels_mfcc'].astype(np.float32)).cpu().detach()
        self.m16 = torch.tensor(dataset[split_type]['vowels_rms'].astype(np.float32)).cpu().detach()
        self.m17 = torch.tensor(dataset[split_type]['vowels_zcr'].astype(np.float32)).cpu().detach()
        self.m18 = torch.tensor(dataset[split_type]['vowels_cenrtoid'].astype(np.float32)).cpu().detach()
        self.m19 = torch.tensor(dataset[split_type]['vowels_log_mel'].astype(np.float32)).cpu().detach()
        self.m20 = torch.tensor(dataset[split_type]['vowels_gfcc'].astype(np.float32)).cpu().detach()
        self.m21 = torch.tensor(dataset[split_type]['vowels_cqcc'].astype(np.float32)).cpu().detach()
        self.m22 = torch.tensor(dataset[split_type]['words_mfcc'].astype(np.float32)).cpu().detach()
        self.m23 = torch.tensor(dataset[split_type]['words_rms'].astype(np.float32)).cpu().detach()
        self.m24 = torch.tensor(dataset[split_type]['words_zcr'].astype(np.float32)).cpu().detach()
        self.m25 = torch.tensor(dataset[split_type]['words_cenrtoid'].astype(np.float32)).cpu().detach()
        self.m26 = torch.tensor(dataset[split_type]['words_log_mel'].astype(np.float32)).cpu().detach()
        self.m27 = torch.tensor(dataset[split_type]['words_gfcc'].astype(np.float32)).cpu().detach()
        self.m28 = torch.tensor(dataset[split_type]['words_cqcc'].astype(np.float32)).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['label'].astype(np.float32)).cpu().detach()
        # Note: this is STILL an numpy array
        self.meta = dataset[split_type]['id']

        self.data = data
        
        self.n_modalities = 28
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.m1.shape[1], self.m2.shape[1], self.m3.shape[1],self.m4.shape[1],self.m5.shape[1], self.m6.shape[1],self.m7.shape[1]
    def get_seq_len1(self):
        return self.m8.shape[1], self.m9.shape[1],self.m10.shape[1],self.m11.shape[1], self.m12.shape[1],self.m13.shape[1],self.m14.shape[1]
    def get_seq_len2(self):
        return self.m15.shape[1], self.m16.shape[1], self.m17.shape[1],self.m18.shape[1],self.m19.shape[1], self.m20.shape[1],self.m21.shape[1]
    def get_seq_len3(self):
        return self.m22.shape[1], self.m23.shape[1],self.m24.shape[1],self.m25.shape[1], self.m26.shape[1],self.m27.shape[1],self.m28.shape[1]
    def get_dim(self):
        return self.m1.shape[2], self.m2.shape[2], self.m3.shape[2],self.m4.shape[2],self.m5.shape[2], self.m6.shape[2],self.m7.shape[2]
    def get_dim1 (self):
        return self.m8.shape[2], self.m9.shape[2],self.m10.shape[2],self.m11.shape[2], self.m12.shape[2],self.m13.shape[2],self.m14.shape[2]
    def get_dim2(self):
        return self.m15.shape[2], self.m16.shape[2], self.m17.shape[2],self.m18.shape[2],self.m19.shape[2], self.m20.shape[2],self.m21.shape[2]
    def get_dim3 (self):
        return self.m22.shape[2], self.m23.shape[2],self.m24.shape[2],self.m25.shape[2], self.m26.shape[2],self.m27.shape[2],self.m28.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.m1[index], self.m2[index], self.m3[index],self.m4[index],self.m5[index], self.m6[index],self.m7[index],
             self.m8[index], self.m9[index],self.m10[index],self.m11[index], self.m12[index],self.m13[index],self.m14[index], 
             self.m15[index], self.m16[index], self.m17[index],self.m18[index],self.m19[index], self.m20[index],self.m21[index],
             self.m22[index], self.m23[index],self.m24[index],self.m25[index], self.m26[index],self.m27[index],self.m28[index])
        Y = self.labels[index]
        META = self.meta[index][0] 
        return X, Y, META        

