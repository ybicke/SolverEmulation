import torch
import pickle
import numpy as np
from os.path import join
from matplotlib import pyplot as plt




models = [  
          
    #{'name': 'AFNO-Emb128-lamb-000','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_000/test'},   
    # {'name': 'AFNO-Emb128-lamb-00000','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_noSparsification/test'},      
    
    #{'name': 'AFNO-Emb128-concat-standard','path': '/mydata/deepcloud/yves/online-datasets/workspace/results/afno_column_1percent_Emb128_easy_concat2/test'},
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
    # {'name': 'AFNO-Emb128_lamb008','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_008/test'},
    #{'name': 'AFNO-Emb128_lamb012','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_012/test'},
    
    
    # {'name': 'AFNO-Emb128_33Freq','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_clean_smooth33/test'},    
    # {'name': 'AFNO-Emb128_34Freq','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_clean_smooth34/test'},    
    # {'name': 'AFNO-Emb128_35Freq','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_clean_smooth35/test'},    
   
    #{'name': 'AFNO-Emb128_cross_clean','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_afno_crossAttention_clean/test'},    
    #{'name': 'AFNO-Emb128_cross_clean1','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_afno_crossAttention_clean1/test'},    
    #{'name': 'AFNO-Emb128_afno_standard','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_test1/test'},    
    
    #{'name': 'AFNO-Emb128','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_clean/test'},   
    
    # # {'name': 'AFNO-Emb128-concat','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_control/test'},   
    # {'name': 'ViT-Emb128-standard','path': '/mydata/deepcloud/yves/results_git/vit_column_1percent_emb128_h4/test'},   

    # {'name': 'AFNO-Emb128-Cross','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_afno_crossAttention_clean/test'},   

    
    # {'name': 'AFNO-Emb128_HeightSpecSig_lwup','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_HeightSpecificSigmoid_lwup/test'},    
    #{'name': 'AFNO-Emb128_sparse008','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_008/test'},    
    #{'name': 'AFNO-Emb128_sparse012','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_012/test'},    
 

    
    # {'name': 'AFNO-Emb128-lamb-004','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_004/test'},
       
    # {'name': 'AFNO-Emb128-lamb-008','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_008/test'},   
    # {'name': 'AFNO-Emb128-lamb-012','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_012/test'},   

    #{'name': 'AFNO-Emb128-lamb-00021','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_00021/test'},   
    # {'name': 'AFNO-Emb128-lamb-00043','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_00043/test'},   
    # {'name': 'AFNO-Emb128-lamb-00106','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_00106/test'},   
    # {'name': 'AFNO-Emb128-lamb-00172','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_00172/test'},  
    # {'name': 'AFNO-Emb128-lamb-00208','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_easyConcat_00208/test'},   
 






    
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


    #{'name': 'AFNO-Emb128-SpecSigmoid','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_HeightSpecificSigmoid_diffNorm/test'},
    #{'name': 'ViT-Emb128-SpecSigmoid','path': '/mydata/deepcloud/yves/results_git/vit_column_1percent_Emb128_HeightSpecificSigmoid_diffNorm/test'},
    #{'name': 'AFNO-Emb128-SpecSig-lwUp','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_HeightSpecificSigmoid_lwup/test'},
    
    

    #{'name': 'AFNO-Emb128-SpecSigmoid-lwDown','path': '/mydata/deepcloud/shared/results-temp/afno_column_1percent_Emb128_HeightSpecificSigmoid_lwdown/test'},
    #{'name': 'AFNO-Emb128-SpecSig-lw-Down','path': '/mydata/deepcloud/shared/results-temp/afno_column_1percent_Emb128_HeightSpecificSigmoid_lwdown_concat/test'},
    #{'name': 'AFNO-Emb128-SpecSig-lw-DownUp','path': '/mydata/deepcloud/shared/results-temp/afno_column_1percent_Emb128_HeightSpecificSigmoid_concat/test'},


    # {'name': 'AFNO-Emb128-concat_w0_layerNorm','path': '/mydata/deepcloud/yves/results_git/afno_column_1percent_Emb128_concatEasy_clean_wO_LayerNorm/test'},
    #{'name': 'GNN-skip-Emb128-l4-h5','path': '/mydata/deepcloud/yves/results_git/graphCast_hirarchical_lrPlateau_00001_l4_h5_bs_512_Emb256/test'},
    #{'name': 'GNN-skip-Emb128-l4-h5-concat','path': '/mydata/deepcloud/yves/results_git/graphCast_hirarchical_lrPlateau_00001_l4_h5_bs_512_Emb256_concat/test'},


    #{'name': 'GNN-gt70-Emb128-l2','path': '/mydata/deepcloud/yves/results_git/graphCast_multiMesh_00001_L2_H70_Emb128_new/test'},
    #{'name': 'GNN-gt70-Emb128-l3','path': '/mydata/deepcloud/yves/results_git/graphCast_multiMesh_00001_L3_H70_Emb128_new/test'},
    # {'name': 'GNN-gt70-Emb256-l4','path': '/mydata/deepcloud/yves/results_git/graphCast_multiMesh_00001_L4_H70_Emb256_new/test'},
    # {'name': 'Bi-LSTM-Emb128','path': '/mydata/deepcloud/yves/results_git/rnn_BiLSTM_128_128/test'},
    #{'name': 'Bi-LSTM-Emb256','path': '/mydata/deepcloud/yves/results_git/rnn_BiLSTM_256_256/test'},
    
    # {'name': 'GNN-Hirarchical-Emb256-l4-h5','path': '/mydata/deepcloud/yves/results_git/graphCast_hirarchical_lrPlateau_00001_l4_h5_bs_512_Emb256_concat/test'},
    #{'name': 'GNN-edgeFeatures-Emb256-l4-h5','path': '/mydata/deepcloud/yves/results_git/graphCast_multiMesh_00001_L3_H70_Emb128_edgeFeatures/test'},
    
    #{'name': 'GNN-256-h5-l4-broadcast-skip','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_256_h5_l4_broadcast_skip/test'},

    #{'name': 'GNN-32-l3','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_small_broadcast/test'},
    #{'name': 'GNN-64-l3','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_64_l3/test'},
    #{'name': 'GNN-64-l2','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_64_l2/test'},
    #{'name': 'GNN-64-l4','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_64_l4/test'},

    #{'name': 'GNN-32-l6','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_32_l6/test'},
    
    #{'name': 'GNN-8-l3','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_8_l3_optimized1/test'},
    #{'name': 'GNN-16-l3','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_16_l3_optimized1/test'},
    {'name': 'GNN-32-l3','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_32_l3_optimized/test'},
    #{'name': 'GNN-32-l3-hrl-02','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_32_l3_hrl_02/test'},
    #{'name': 'GNN-32-l3-hrl-05','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_32_l3_hrl_05/test'},
    # {'name': 'GNN-32-l3-hrl-1','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_32_l3_hrl_1/test'},

    {'name': 'GNN-32-l3-hrl-005','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_32_l3_hrl_005/test'},
    {'name': 'GNN-32-l3-hrlu-005','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_32_l3_hrlu_005/test'},


    #{'name': 'GNN-64-l3','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_64_l3_optimized/test'},
    #{'name': 'GNN-128-l3','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_128_l3_optimized/test'},


    
    
    
    
    #{'name': 'GNN-512-l3','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/gnn_medium/test'},
    #{'name': 'BiLSTM-128-265','path': '/mydata/deepcloud/yves/A_RadiativeFlux/results/rnn_medium/test'},


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


    # Select a specific sample index
    #sample_index = (0,5)  # Change this to the desired sample index
    #y_true = y_true[sample_index]
    #y_pred = y_pred[sample_index]
    #h_true = h_true[sample_index]
    #h_pred = h_pred[sample_index]

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

plt.savefig('/mydata/deepcloud/yves/GNN_Comparing_HRLU_Models.png', bbox_inches='tight', dpi=300)
