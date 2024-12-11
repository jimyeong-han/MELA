# parameters
result_dir="result/refinement/"

TOKENIZERS_PARALLELISM=false
exp_name='star_UniMD_ft_sr_event_list_0.4_orig_ans_e_loc'
ckpt='lavis/result/finetune/star_ft_origin/checkpoint_best.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 29600 train.py \
--cfg-path lavis/projects/sevila/train/star.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=4 \
datasets.star.vis_processor.train.n_frms=8 \
datasets.star.vis_processor.eval.n_frms=32 \
run.batch_size_train=16 \
run.batch_size_eval=12 \
run.init_lr=3e-5 \
run.max_epoch=15 \
run.warmup_steps=500 \
run.accum_grad_iters=1 \
model.task='train_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'