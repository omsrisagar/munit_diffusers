export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES='0, 1, 2, 3' # MUST GIVE SPACES for NUM_GPUS below to be correct
#export CUDA_VISIBLE_DEVICES='2, 1, 3'
#export CUDA_VISIBLE_DEVICES='0, 1'
export CUDA_VISIBLE_DEVICES='0'
export ROOT='sketch2image/stage2_decoder/'
export LOGDIR=sketch2image/stage2_decoder/inference/text1/
export PYTHONPATH=$PYTHONPATH:$(pwd)
CVD=($CUDA_VISIBLE_DEVICES)
NUM_GPUS=${#CVD[@]}
echo $NUM_GPUS
MODEL_FLAGS="--learn_sigma False --uncond_p 0 --image_size 256"
TRAIN_FLAGS="--batch_size 1 --resume_checkpoint ROOT/LATEST"
DIFFUSION_FLAGS="--use_kl False --rescale_learned_sigmas False --beta_start 0.00085 --beta_end 0.012"
SAMPLE_FLAGS="--num_samples 4 --sample_c 7.5 --img_disp_nrow 1"
DATASET_FLAGS="--data_dir /export/home/cuda00022/srikanth/Projects/munit_diffusers/examples/obayashi/data/inference/infer_img/ --mode edge"
mpiexec -n $NUM_GPUS python ./image_sample.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
# python ./image_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS  $DATASET_FLAGS
