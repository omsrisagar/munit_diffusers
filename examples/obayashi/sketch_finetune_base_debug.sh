#export CUDA_LAUNCH_BLOCKING=1
##export CUDA_VISIBLE_DEVICES='0, 1, 2, 3' # MUST GIVE SPACES for NUM_GPUS below to be correct
#export CUDA_VISIBLE_DEVICES='2, 1, 3'
##export CUDA_VISIBLE_DEVICES='0'
#export LOGDIR=sketch2image/stage1/
#export PYTHONPATH=$PYTHONPATH:$(pwd)
#CVD=($CUDA_VISIBLE_DEVICES)
#NUM_GPUS=${#CVD[@]}
#echo $NUM_GPUS
#MODEL_FLAGS="--learn_sigma False --uncond_p 0. --image_size 256 --finetune_decoder False"
##TRAIN_FLAGS="--lr 3.5e-5 --batch_size 1  --schedule_sampler loss-second-moment  --model_path ./ckpt/base.pt
#TRAIN_FLAGS="--lr 7.0e-5 --batch_size 32  --schedule_sampler loss-second-moment
#--lr_anneal_steps 200000"
#DIFFUSION_FLAGS="--use_kl False --rescale_learned_sigmas False --beta_start 0.00085 --beta_end 0.012"
#SAMPLE_FLAGS="--num_samples 48 --sample_c 7.5 --img_disp_nrow 16" # in one forward pass batch_size/2* num_gpus are generated
##DATASET_FLAGS="--data_dir ./dataset/COCOSTUFF_val.txt --val_data_dir ./dataset/COCOSTUFF_val.txt --mode coco-edge"
##DATASET_FLAGS="--data_dir from_unsplash/images-facade/images-facade_img/ --mode edge"
#DATASET_FLAGS="--data_dir /export/home/cuda00022/srikanth/Projects/munit_diffusers/examples/obayashi/data/train_img/ --mode edge"
##mpiexec -n $NUM_GPUS --allow-run-as-root python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
#mpiexec -n $NUM_GPUS python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
## python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
#exit 0


export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES='0, 1, 2, 3' # MUST GIVE SPACES for NUM_GPUS below to be correct
#export CUDA_VISIBLE_DEVICES='2, 1, 3'
export CUDA_VISIBLE_DEVICES='0'
export LOGDIR=sketch2image/stage1_cont_debug/
export PYTHONPATH=$PYTHONPATH:$(pwd)
CVD=($CUDA_VISIBLE_DEVICES)
NUM_GPUS=${#CVD[@]}
echo $NUM_GPUS
MODEL_FLAGS="--learn_sigma False --uncond_p 0.2 --image_size 256 --finetune_decoder False --encoder_path
./sketch2image/stage1/checkpoints/ema_0.9999_190000.pt"
#TRAIN_FLAGS="--lr 5e-5 --batch_size 2 --schedule_sampler loss-second-moment  --model_path ./ckpt/base.pt
TRAIN_FLAGS="--lr 5e-5 --batch_size 2 --schedule_sampler loss-second-moment
--lr_anneal_steps 2"
DIFFUSION_FLAGS="--use_kl False --rescale_learned_sigmas False --beta_start 0.00085 --beta_end 0.012"
SAMPLE_FLAGS="--num_samples 2 --sample_c 7.5 --img_disp_nrow 2"
DATASET_FLAGS="--data_dir /export/home/cuda00022/srikanth/Projects/munit_diffusers/examples/obayashi/data/train_img/ --mode edge"
mpiexec -n $NUM_GPUS python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
# python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
exit 0


export LOGDIR=./coco-edge/coco-64-stage2-decoder/
export PYTHONPATH=$PYTHONPATH:$(pwd)
NUM_GPUS=1
MODEL_FLAGS="--learn_sigma True --uncond_p 0.2 --image_size 64 --finetune_decoder True"
TRAIN_FLAGS="--lr 3.5e-5 --batch_size 2 --schedule_sampler loss-second-moment --model_path ./ckpt/base.pt
--encoder_path ./coco-edge/coco-64-stage1-cont/checkpoints/ema_0.9999_000000.pt"
DIFFUSION_FLAGS=""
SAMPLE_FLAGS="--num_samples 2 --sample_c 2.5"
DATASET_FLAGS="--data_dir ./dataset/COCOSTUFF_val.txt --val_data_dir ./dataset/COCOSTUFF_val.txt --mode coco-edge"
mpiexec -n $NUM_GPUS --allow-run-as-root python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
# python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
 

