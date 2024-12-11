# parameters
result_dir="result/finetune/"

exp_name='star_UniMD_ft_event_list_0.4_char2star'
#exp_name="test"
ckpt='mela_checkpoints/mela_pretrained.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/mela/train/star.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=4 \
datasets.star.vis_processor.train.n_frms=32 \
datasets.star.vis_processor.eval.n_frms=32 \
run.batch_size_train=8 \
run.batch_size_eval=8 \
run.init_lr=3e-5 \
run.max_epoch=10 \
run.warmup_steps=1000 \
run.accum_grad_iters=2 \
model.task='qvh_freeze_loc_train_qa_with_loc_train_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'