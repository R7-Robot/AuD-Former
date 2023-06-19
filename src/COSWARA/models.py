import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder

#GAT
class AUDFORMERModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(AUDFORMERModel, self).__init__()
        self.orig_d_m1, self.orig_d_m2, self.orig_d_m3, self.orig_d_m4 = hyp_params.orig_d_m1, hyp_params.orig_d_m2, hyp_params.orig_d_m3, hyp_params.orig_d_m4
        self.orig_d_m5, self.orig_d_m6, self.orig_d_m7, self.orig_d_m8 = hyp_params.orig_d_m5, hyp_params.orig_d_m6, hyp_params.orig_d_m7, hyp_params.orig_d_m8
        self.orig_d_m9, self.orig_d_m10, self.orig_d_m11, self.orig_d_m12 = hyp_params.orig_d_m9, hyp_params.orig_d_m10, hyp_params.orig_d_m11, hyp_params.orig_d_m12
        self.orig_d_m13, self.orig_d_m14,self.orig_d_m15, self.orig_d_m16 = hyp_params.orig_d_m13, hyp_params.orig_d_m14,hyp_params.orig_d_m15, hyp_params.orig_d_m16
        self.orig_d_m17, self.orig_d_m18,self.orig_d_m19, self.orig_d_m20 = hyp_params.orig_d_m17, hyp_params.orig_d_m18,hyp_params.orig_d_m19, hyp_params.orig_d_m20
        self.orig_d_m21, self.orig_d_m22,self.orig_d_m23, self.orig_d_m24 = hyp_params.orig_d_m21, hyp_params.orig_d_m22,hyp_params.orig_d_m23, hyp_params.orig_d_m24
        self.orig_d_m25, self.orig_d_m26,self.orig_d_m27, self.orig_d_m28 = hyp_params.orig_d_m25, hyp_params.orig_d_m26,hyp_params.orig_d_m27, hyp_params.orig_d_m28
        self.d_m = 30

        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        combined_dim = 30     
        output_dim = hyp_params.output_dim        # This is actually not a hyperparameter :-)
        self.channels = (hyp_params.m1_len+hyp_params.m2_len+hyp_params.m3_len+hyp_params.m4_len+hyp_params.m5_len+hyp_params.m6_len+hyp_params.m7_len +
                        hyp_params.m8_len+hyp_params.m9_len+hyp_params.m10_len+hyp_params.m11_len+hyp_params.m12_len+hyp_params.m13_len+hyp_params.m14_len +
                        hyp_params.m15_len+ hyp_params.m16_len+ hyp_params.m17_len+ hyp_params.m18_len+ hyp_params.m19_len+ hyp_params.m20_len+ hyp_params.m21_len +
                        hyp_params.m22_len+ hyp_params.m23_len+ hyp_params.m24_len+ hyp_params.m25_len+ hyp_params.m26_len+ hyp_params.m27_len+ hyp_params.m28_len )
        
        self.channels_1 = hyp_params.m1_len+hyp_params.m2_len+hyp_params.m3_len+hyp_params.m4_len+hyp_params.m5_len+hyp_params.m6_len+hyp_params.m7_len
        self.channels_2 = hyp_params.m8_len+hyp_params.m9_len+hyp_params.m10_len+hyp_params.m11_len+hyp_params.m12_len+hyp_params.m13_len+hyp_params.m14_len
        self.channels_3 = hyp_params.m15_len+ hyp_params.m16_len+ hyp_params.m17_len+ hyp_params.m18_len+ hyp_params.m19_len+ hyp_params.m20_len+ hyp_params.m21_len
        self.channels_4 = hyp_params.m22_len+ hyp_params.m23_len+ hyp_params.m24_len+ hyp_params.m25_len+ hyp_params.m26_len+ hyp_params.m27_len+ hyp_params.m28_len 

        # 1. Temporal convolutional layers
        self.proj_m1 = nn.Conv1d(self.orig_d_m1, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m2 = nn.Conv1d(self.orig_d_m2, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m3 = nn.Conv1d(self.orig_d_m3, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m4 = nn.Conv1d(self.orig_d_m4, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m5 = nn.Conv1d(self.orig_d_m5, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m6 = nn.Conv1d(self.orig_d_m6, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m7 = nn.Conv1d(self.orig_d_m7, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m8 = nn.Conv1d(self.orig_d_m8, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m9 = nn.Conv1d(self.orig_d_m9, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m10 = nn.Conv1d(self.orig_d_m10, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m11 = nn.Conv1d(self.orig_d_m11, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m12 = nn.Conv1d(self.orig_d_m12, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m13 = nn.Conv1d(self.orig_d_m13, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m14 = nn.Conv1d(self.orig_d_m14, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m15 = nn.Conv1d(self.orig_d_m15, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m16 = nn.Conv1d(self.orig_d_m16, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m17 = nn.Conv1d(self.orig_d_m17, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m18 = nn.Conv1d(self.orig_d_m18, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m19 = nn.Conv1d(self.orig_d_m19, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m20 = nn.Conv1d(self.orig_d_m20, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m21 = nn.Conv1d(self.orig_d_m21, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m22 = nn.Conv1d(self.orig_d_m22, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m23 = nn.Conv1d(self.orig_d_m23, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m24 = nn.Conv1d(self.orig_d_m24, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m25 = nn.Conv1d(self.orig_d_m25, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m26 = nn.Conv1d(self.orig_d_m26, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m27 = nn.Conv1d(self.orig_d_m27, self.d_m, kernel_size=1, padding=0, bias=False)
        self.proj_m28 = nn.Conv1d(self.orig_d_m28, self.d_m, kernel_size=1, padding=0, bias=False)

        self.final_conv = nn.Conv1d(self.channels, 1, kernel_size=1, padding=0, bias=False)
        self.final_lin = nn.Linear(self.channels,1)
        # 2. global attention
        self.trans_m1_all = self.get_network(self_type='m1_all', layers=3)
        self.trans_m2_all = self.get_network(self_type='m2_all', layers=3)
        self.trans_m3_all = self.get_network(self_type='m3_all', layers=3)
        self.trans_m4_all = self.get_network(self_type='m4_all', layers=3)
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        self.trans_final = self.get_network(self_type='policy', layers=5)
        self.trans_self = self.get_network(self_type='self', layers=1)

        # Projection layers
        self.proj1 = self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['m1_all','m2_all','m3_all','m4_all','self','policy']:
            embed_dim, attn_dropout = self.d_m, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    def forward(self,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,m21,m22,m23,m24,m25,m26,m27,m28):

        m_1 = m1.transpose(1, 2)
        m_2 = m2.transpose(1, 2)
        m_3 = m3.transpose(1, 2)
        m_4 = m4.transpose(1, 2)
        m_5 = m5.transpose(1, 2)
        m_6 = m6.transpose(1, 2)
        m_7 = m7.transpose(1, 2)
        m_8 = m8.transpose(1, 2)
        m_9 = m9.transpose(1, 2)
        m_10 = m10.transpose(1, 2)
        m_11 = m11.transpose(1, 2)
        m_12 = m12.transpose(1, 2)
        m_13 = m13.transpose(1, 2)
        m_14 = m14.transpose(1, 2)
        m_15 = m15.transpose(1, 2)
        m_16 = m16.transpose(1, 2)
        m_17 = m17.transpose(1, 2)
        m_18 = m18.transpose(1, 2)
        m_19 = m19.transpose(1, 2)
        m_20 = m20.transpose(1, 2)
        m_21 = m21.transpose(1, 2)
        m_22 = m22.transpose(1, 2)
        m_23 = m23.transpose(1, 2)
        m_24 = m24.transpose(1, 2)
        m_25 = m25.transpose(1, 2)
        m_26 = m26.transpose(1, 2)
        m_27 = m27.transpose(1, 2)
        m_28 = m28.transpose(1, 2)
        # Project features
        proj_x_m1 = m_1 if self.orig_d_m1 == self.d_m else self.proj_m1(m_1)
        proj_x_m2 = m_2 if self.orig_d_m2 == self.d_m else self.proj_m2(m_2)
        proj_x_m3 = m_3 if self.orig_d_m3 == self.d_m else self.proj_m3(m_3)
        proj_x_m4 = m_4 if self.orig_d_m4 == self.d_m else self.proj_m4(m_4)
        proj_x_m5 = m_5 if self.orig_d_m5 == self.d_m else self.proj_m5(m_5)
        proj_x_m6 = m_6 if self.orig_d_m6 == self.d_m else self.proj_m6(m_6)
        proj_x_m7 = m_7 if self.orig_d_m7 == self.d_m else self.proj_m7(m_7)
        proj_x_m8 = m_8 if self.orig_d_m8 == self.d_m else self.proj_m8(m_8)
        proj_x_m9 = m_9 if self.orig_d_m9 == self.d_m else self.proj_m9(m_9)
        proj_x_m10 = m_10 if self.orig_d_m10 == self.d_m else self.proj_m10(m_10)
        proj_x_m11 = m_11 if self.orig_d_m11 == self.d_m else self.proj_m11(m_11)
        proj_x_m12 = m_12 if self.orig_d_m12 == self.d_m else self.proj_m12(m_12)
        proj_x_m13 = m_13 if self.orig_d_m13 == self.d_m else self.proj_m13(m_13)
        proj_x_m14 = m_14 if self.orig_d_m14 == self.d_m else self.proj_m14(m_14)
        proj_x_m15 = m_15 if self.orig_d_m15 == self.d_m else self.proj_m15(m_15)
        proj_x_m16 = m_16 if self.orig_d_m16 == self.d_m else self.proj_m16(m_16)
        proj_x_m17 = m_17 if self.orig_d_m17 == self.d_m else self.proj_m17(m_17)
        proj_x_m18 = m_18 if self.orig_d_m18 == self.d_m else self.proj_m18(m_18)
        proj_x_m19 = m_19 if self.orig_d_m19 == self.d_m else self.proj_m19(m_19)
        proj_x_m20 = m_20 if self.orig_d_m20 == self.d_m else self.proj_m20(m_20)
        proj_x_m21 = m_21 if self.orig_d_m21 == self.d_m else self.proj_m21(m_21)
        proj_x_m22 = m_23 if self.orig_d_m22 == self.d_m else self.proj_m22(m_22)
        proj_x_m23 = m_23 if self.orig_d_m23 == self.d_m else self.proj_m23(m_23)
        proj_x_m24 = m_24 if self.orig_d_m24 == self.d_m else self.proj_m24(m_24)
        proj_x_m25 = m_25 if self.orig_d_m25 == self.d_m else self.proj_m25(m_25)
        proj_x_m26 = m_26 if self.orig_d_m26 == self.d_m else self.proj_m26(m_26)
        proj_x_m27 = m_27 if self.orig_d_m27 == self.d_m else self.proj_m27(m_27)
        proj_x_m28 = m_28 if self.orig_d_m28 == self.d_m else self.proj_m28(m_28)

        proj_x_m1 = proj_x_m1.permute(2, 0, 1)
        proj_x_m2 = proj_x_m2.permute(2, 0, 1)
        proj_x_m3 = proj_x_m3.permute(2, 0, 1)
        proj_x_m4 = proj_x_m4.permute(2, 0, 1)
        proj_x_m5 = proj_x_m5.permute(2, 0, 1)
        proj_x_m6 = proj_x_m6.permute(2, 0, 1)
        proj_x_m7 = proj_x_m7.permute(2, 0, 1)
        proj_x_m8 = proj_x_m8.permute(2, 0, 1)
        proj_x_m9 = proj_x_m9.permute(2, 0, 1)
        proj_x_m10 = proj_x_m10.permute(2, 0, 1)
        proj_x_m11 = proj_x_m11.permute(2, 0, 1)
        proj_x_m12 = proj_x_m12.permute(2, 0, 1)
        proj_x_m13 = proj_x_m13.permute(2, 0, 1)
        proj_x_m14 = proj_x_m14.permute(2, 0, 1)
        proj_x_m15 = proj_x_m15.permute(2, 0, 1)
        proj_x_m16 = proj_x_m16.permute(2, 0, 1)
        proj_x_m17 = proj_x_m17.permute(2, 0, 1)
        proj_x_m18 = proj_x_m18.permute(2, 0, 1)
        proj_x_m19 = proj_x_m19.permute(2, 0, 1)
        proj_x_m20 = proj_x_m20.permute(2, 0, 1)
        proj_x_m21 = proj_x_m21.permute(2, 0, 1)
        proj_x_m22 = proj_x_m22.permute(2, 0, 1)
        proj_x_m23 = proj_x_m23.permute(2, 0, 1)
        proj_x_m24 = proj_x_m24.permute(2, 0, 1)
        proj_x_m25 = proj_x_m25.permute(2, 0, 1)
        proj_x_m26 = proj_x_m26.permute(2, 0, 1)
        proj_x_m27 = proj_x_m27.permute(2, 0, 1)
        proj_x_m28 = proj_x_m28.permute(2, 0, 1)
        
        proj_1_ = torch.cat([proj_x_m1 , proj_x_m2 , proj_x_m3 , proj_x_m4, proj_x_m5 , proj_x_m6 , proj_x_m7] , dim = 0)
        proj_2_ = torch.cat([proj_x_m8 , proj_x_m9 , proj_x_m10 , proj_x_m11, proj_x_m12 , proj_x_m13 , proj_x_m14] , dim = 0)
        proj_3_ = torch.cat([proj_x_m15 , proj_x_m16 , proj_x_m17 , proj_x_m18, proj_x_m19 , proj_x_m20 , proj_x_m21] , dim = 0)
        proj_4_ = torch.cat([proj_x_m22 , proj_x_m23 , proj_x_m24 , proj_x_m25, proj_x_m26 , proj_x_m27 , proj_x_m28] , dim = 0)

        proj_1 = self.trans_self(proj_1_)
        proj_2 = self.trans_self(proj_2_)
        proj_3 = self.trans_self(proj_3_)
        proj_4 = self.trans_self(proj_4_)

        proj_all = torch.cat([proj_1,proj_2,proj_3,proj_4] , dim = 0)

        p1_with_all = self.trans_m1_all(proj_1,proj_all,proj_all)
        p2_with_all = self.trans_m2_all(proj_2,proj_all,proj_all)
        p3_with_all = self.trans_m3_all(proj_3,proj_all,proj_all)
        p4_with_all = self.trans_m4_all(proj_4,proj_all,proj_all)

        last_hs1 = torch.cat([p1_with_all,p2_with_all,p3_with_all,p4_with_all] , dim = 0)
        last_hs2 = self.trans_final(last_hs1)
        last_hs = self.final_lin(last_hs2.permute(1,2,0)).squeeze(-1)

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs)

        return output, last_hs
