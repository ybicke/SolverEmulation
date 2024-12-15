import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt




models = [  
          
    # {'name': 'AFNO-Emb128_easy_concat','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_128_easy_concat/test'},
    #{'name': 'AFNO-Emb128_easy_concat_2','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_128_easy_concat2/test'},    
    #{'name': 'AFNO-Emb128_concat_mlp','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_128_easy_concat_mlp/test'},
    #{'name': 'AFNO-Emb128_crossAttention','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_128_crossAttention/test'},    

        
    #{'name': 'AFNO-Emb128_lowpass_001','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_lowpass_001/test'},
    #{'name': 'AFNO-Emb128_lowpass_002','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_lowpass_002/test'},
    #{'name': 'AFNO-Emb128_lowpass_0005','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_lowpass_0005/test'},
    
    #{'name': 'AFNO-Emb128_sparse004','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_spars004/test'},
    #{'name': 'AFNO-Emb128_sparse006','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_spars006/test'},
    #{'name': 'AFNO-Emb128_sparse008','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_spars008/test'},

    #{'name': 'AFNO-Emb128_sparse0001','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat/test'},    
    #{'name': 'AFNO-Emb128_sparse004','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_004/test'},
    #{'name': 'AFNO-Emb128_sparse008','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_008/test'},
    #{'name': 'AFNO-Emb128_sparse012','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_012/test'},
    #{'name': 'AFNO-Emb128_sparse000','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_noSparsification/test'},
    
    
    #{'name': 'AFNO-Emb128_smooth33','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_clean_smooth33/test'},    
    #{'name': 'AFNO-Emb128_smooth33','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_clean_smooth34/test'},    
    #{'name': 'AFNO-Emb128_smooth33','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_clean_smooth35/test'},    
   
    #{'name': 'AFNO-Emb128_cross_clean','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_afno_crossAttention_clean/test'},    
    #{'name': 'AFNO-Emb128_cross_clean1','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_afno_crossAttention_clean1/test'},    
    #{'name': 'AFNO-Emb128_afno_standard','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_test1/test'},    
    
    {'name': 'AFNO-Emb128_clean','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_clean/test'},   
    {'name': 'AFNO-Emb128_HeightSpecSig_lwup','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_HeightSpecificSigmoid_lwup/test'},    
    #{'name': 'AFNO-Emb128_sparse008','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_008/test'},    
    #{'name': 'AFNO-Emb128_sparse012','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_012/test'},    
 




    
    #{'name': 'AFNO-Emb128_hard06','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_hard06/test'},
    #{'name': 'AFNO-Emb128_hard07','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_hard07/test'},
    #{'name': 'AFNO-Emb128_hard08','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_hard08/test'},
    #{'name': 'AFNO-Emb128_hard09','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_hard09/test'},
    
    #{'name': 'AFNO-Emb128_spars02','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_spars02/test'},
    #{'name': 'AFNO-Emb128_spars03','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_spars03/test'},
    #{'name': 'AFNO-Emb128_spars04','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_spars04/test'},
    #{'name': 'AFNO-Emb128_spars05','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_spars05/test'},
    
    #{'name': 'ViT-Emb128_l3','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/vit_column_1percent_emb128_h3/test'},
    #{'name': 'ViT-Emb128_l4','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/vit_column_1percent_emb128_h4/test'},
    #{'name': 'ViT-Emb128_l5','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/vit_column_1percent_emb128_h5/test'},
    #{'name': 'ViT-Emb128_l6','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/vit_column_1percent_emb128_h6/test'},
    
    #{'name': 'AFNO-Emb128_l3','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_h3/test'},
    #{'name': 'AFNO-Emb128_l4','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_h4/test'},
    #{'name': 'AFNO-Emb128_l5','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_h5/test'},
    #{'name': 'AFNO-Emb128_l6','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_h6/test'},
    
    #{'name': 'ViT-Emb8','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/vit_column_1percent_Emb8/test'},
    #{'name': 'ViT-Emb16','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/vit_column_1percent_Emb16/test'},
    #{'name': 'ViT-Emb32','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/vit_column_1percent_Emb32/test'},
    #{'name': 'ViT-Emb64','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/vit_column_1percent_Emb64/test'},
    
    #{'name': 'AFNO-Emb8','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb8/test'},
    #{'name': 'AFNO-Emb16','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb16/test'},
    #{'name': 'AFNO-Emb32','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb32/test'},
    #{'name': 'AFNO-Emb64','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb64/test'},
    
    #{'name': 'AFNO-Test_year','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent/test'},
    #{'name': 'ViT-Test_year', 'path': '/mydata/deepcloud/yves/online-datasets/workspace/results/vit_column_1year_30percent/test_year_checkpoint'},
]


y_mae_gs, y_mae_hs, h_mae_gs, h_mae_hs = list(), list(), list(), list()
for model in models:
    test_path = model['path']

    print(f'loading test files... ({test_path})')
    with open(join(test_path, 'y_true.pickle'), 'rb') as handle:
        y_true = pickle.load(handle)
    print(f'y_true: {y_true.shape}')
    with open(join(test_path, 'y_pred.pickle'), 'rb') as handle:
        y_pred = pickle.load(handle)
    print(f'y_pred: {y_pred.shape}')
    with open(join(test_path, 'h_true.pickle'), 'rb') as handle:
        h_true = pickle.load(handle)
    print(f'h_true: {h_true.shape}')
    with open(join(test_path, 'h_pred.pickle'), 'rb') as handle:
        h_pred = pickle.load(handle)
    print(f'h_pred: {h_pred.shape}')

    print('calculating errors...')
    # y_mae_g = torch.mean(torch.abs(y_true - y_pred), dim=0)
    # h_mae_g = torch.mean(torch.abs(h_true - h_pred), dim=0)
    dim = [0, 1] if len(y_true.shape) == 4 else 0
    y_mae_h = torch.mean(torch.abs(y_true - y_pred), dim=dim)
    h_mae_h = torch.mean(torch.abs(h_true - h_pred), dim=dim)

    # y_mae_gs.append(torch.flip(y_mae_g, [1]))
    # h_mae_gs.append(torch.flip(h_mae_g, [1]))
    y_mae_hs.append(torch.flip(y_mae_h, [0]))
    h_mae_hs.append(torch.flip(h_mae_h, [0]))


def add_supplot(fig, x, ys, id, models_name, xlabel=None, ylabel=None, 
            title=None, log_scale=True, mask=None):
    ax = fig.add_subplot(*id)
    for y, model_name, msk in zip(ys, models_name, mask):
        if msk:
            ax.plot(y, x, label=f'{model_name}')
    ax.grid()
    ax.invert_yaxis()
    ax.set_ylabel(ylabel if ylabel else '')
    ax.set_xlabel(xlabel if xlabel else '')
    ax.set_title(title if title else '')
    if log_scale:
        ax.set_xscale('log')
    return ax
    

# err = np.load('/mydata/deepcloud/salman/predictions/model161_rnn_loss_ecl_v1_norm_pf_scaled_out_optimal_beta.npy')

# RNN Loss
#y_errs = np.load('/mydata/deepcloud/salman/new_result/model161_rnn_loss_ecl_v1_norm_pf_scaled_out_optimal_beta/test/y_errs.npy')
#h_errs = np.load('/mydata/deepcloud/salman/new_result/model161_rnn_loss_ecl_v1_norm_pf_scaled_out_optimal_beta/test/h_errs.npy')

#y_mae_hs.append(y_errs)
#h_mae_hs.append(h_errs)


# ### Plot
models_name = [model['name'] for model in models]
# models_name.append('RNN')

# print(y_errs.shape)
mask = [True] * len(models_name)
fig = plt.figure(figsize=(12, 16))
ax00 = add_supplot(fig, x=range(71), ys=[y[:, 3] for y in y_mae_hs], id=(2, 3, 1), models_name=models_name, title='Downward fluxed', ylabel='Shortwave', xlabel='MAE [W/m$^2$]', mask=mask)
add_supplot(fig, x=range(71), ys=[y[:, 2] for y in y_mae_hs], id=(2, 3, 2), models_name=models_name, title='Upward fluxed', xlabel='MAE [W/m$^2$]', mask=mask)
add_supplot(fig, x=range(70), ys=[h[:, 1] for h in h_mae_hs], id=(2, 3, 3), models_name=models_name, title='Heating rates', xlabel='MAE [K/day]', mask=mask)
add_supplot(fig, x=range(71), ys=[y[:, 1] for y in y_mae_hs], id=(2, 3, 4), models_name=models_name, ylabel='Longwave', xlabel='MAE [W/m$^2$]', mask=mask)
ax0 = add_supplot(fig, x=range(71), ys=[y[:, 0] for y in y_mae_hs], id=(2, 3, 5), models_name=models_name, xlabel='MAE [W/m$^2$]', mask=mask)
add_supplot(fig, x=range(70), ys=[h[:, 0] for h in h_mae_hs], id=(2, 3, 6), models_name=models_name, xlabel='MAE [K/day]', mask=mask)
ax00.legend(fontsize="10", loc='lower left')

plt.savefig('/mydata/deepcloud/yves/results_git/heightSpecSig_lwup.png', bbox_inches='tight', dpi=300)
