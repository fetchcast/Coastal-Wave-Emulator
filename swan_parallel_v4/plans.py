"""Fixed baseline settings; no new hyperparameter search."""
import copy
BASELINES=('convnext_lstm','conv_swin','unet_lstm','u_ffno','swin','convlstm','vit')
SETTINGS={
 'convnext_lstm':dict(convnext_dims=[96,192,384],convnext_depths=[2,2,2],lstm_hidden=256,use_checkpoint=True),
 'conv_swin':dict(base_width=48,swin_dim=256,swin_depth=6,swin_heads=8,window_size=8,use_checkpoint=True),
 'unet_lstm':dict(unet_feat=[64,128,256,512,1024],hidden_dim=768),
 'u_ffno':dict(unet_feat=[128,256,512,1024,2048],hidden_dim=256,fno_width=256,fno_depth=4,modes_x=16,modes_y=16),
 'swin':dict(embed_dim=72,swin_depths=[2,2,6,2],swin_num_heads=[3,6,12,24],patch_size=4,window_size=8,use_checkpoint=True),
 'convlstm':dict(width=128,hidden_dim=128,depth=2),
 'vit':dict(embed_dim=384,vit_depth=6,vit_heads=6,patch_size=16,use_checkpoint=True)}
def jobs(base):
    result=[]
    for seed in (42,43,44):
        for model in BASELINES:
            j=copy.deepcopy(base)
            j.update(model=model,seed=seed,config_id=f'{model}_fixed_v4_s{seed}',stage='pilot',epochs=30,
                     early_stop_patience=0,train_fraction=1.,use_bnd='on')
            j.pop('max_updates',None);j.pop('eval_every_updates',None)
            j['hyperparams']=dict(seq_length=12,hidden_dim=256,fno_width=64,fno_depth=4,modes_x=24,
                modes_y=24,modes_t=4,batch_size=1,acc_steps=4,max_lr=1e-4,weight_decay=1e-4)
            j['hyperparams'].update(SETTINGS[model]);result.append(j)
    return result
