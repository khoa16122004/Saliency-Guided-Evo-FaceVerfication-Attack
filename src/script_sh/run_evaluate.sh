python script/evaluate_result.py \
    --result_dir output/seed\=22520691_log_GA_niter\=1000_label\=0_reconsw\=0.5_attackw\=0.5_saliencyw\=0.0_guided\=0_popsize\=100_toursize\=4_patchsize\=16_problocationmutate\=0.2_probpatchmutate\=0.9_fitnesstype\=normal \
    --algorithm GA_normal 

python script/evaluate_result.py \
    --result_dir output/seed\=22520691_log_GA_niter\=1000_label\=1_reconsw\=0.5_attackw\=0.5_saliencyw\=0.0_guided\=0_popsize\=100_toursize\=4_patchsize\=16_problocationmutate\=0.2_probpatchmutate\=0.9_fitnesstype\=normal \
    --algorithm GA_normal 

python script/evaluate_result.py \
    --result_dir output/seed\=22520691_log_GA_niter\=1000_label\=0_reconsw\=0.5_attackw\=0.5_saliencyw\=0.0_guided\=0_popsize\=100_toursize\=4_patchsize\=16_problocationmutate\=0.2_probpatchmutate\=0.9_fitnesstype\=adaptive \
    --algorithm GA_adaptive

python script/evaluate_result.py \
    --result_dir output/seed\=22520691_log_GA_niter\=1000_label\=1_reconsw\=0.5_attackw\=0.5_saliencyw\=0.0_guided\=0_popsize\=100_toursize\=4_patchsize\=16_problocationmutate\=0.2_probpatchmutate\=0.9_fitnesstype\=adaptive \
    --algorithm GA_adaptive 





