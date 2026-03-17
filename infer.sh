LOCAL_DIR=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/Qwen2-Audio-finetune
cd $LOCAL_DIR

MODEL_PATH=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2-Audio
# TRIAN_DATA_PATH=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/multitask_asr/train
# EVAL_DATA_PATH=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/multitask_asr/dev
TRIAN_DATA_PATH=data/aishell-1/asr/test
EVAL_DATA_PATH=data/aishell-1/asr/test
TRAIN_STRATEGY=deepspeed # ddp deepspeed
DEVICE_TYPE=npu # npu cuda
PEFT_PATH=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/Qwen2-Audio-finetune/exp/15-00-13/15-05/
#parameters
num_workers=8
prefetch_factor=2
python \
    ./infer.py \
    ++env.device_type=$DEVICE_TYPE \
    ++env.model_path=$MODEL_PATH \
    ++data.eval_data_path=$EVAL_DATA_PATH \
    ++data.num_workers=$num_workers \
    ++data.prefetch_factor=$prefetch_factor \
    ++eval.peft_path=$PEFT_PATH

