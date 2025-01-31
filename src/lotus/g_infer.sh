
export CUDA=0

export CHECKPOINT_DIR="jingheya/lotus-depth-g-v2-0-disparity"
export OUTPUT_DIR="output/Depth_G_Infer"
export TASK_NAME="depth"

#export CHECKPOINT_DIR="jingheya/lotus-depth-d-v2-0-disparity "
#export OUTPUT_DIR="output/Depth_D_Infer"
#export TASK_NAME="depth"

#export MODE="regression"
export MODE="generation"

export TEST_IMAGES="../jpgs/dashcam"

CUDA_VISIBLE_DEVICES=$CUDA python infer.py \
        --pretrained_model_name_or_path=$CHECKPOINT_DIR \
        --prediction_type="sample" \
        --seed=42 \
        --half_precision \
        --input_dir=$TEST_IMAGES \
        --task_name=$TASK_NAME \
        --mode=$MODE \
        --output_dir=$OUTPUT_DIR \
        --disparity
