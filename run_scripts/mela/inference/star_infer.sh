# parameters/data path
result_dir="result/zero-shot/"

exp_name='calc_params'
ckpt='lavis/result/finetune/star_UniMD_ft_event_list_0.4_char2star_trained_event_loc/checkpoint_best.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 evaluate.py \
--cfg-path lavis/projects/mela/eval/star_eval.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=4 \
datasets.star.vis_processor.eval.n_frms=32 \
run.batch_size_eval=8 \
model.task='train_loc_train_qa_with_loc_train_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'

#model.task='qvh_freeze_loc_freeze_qa_vid' \
