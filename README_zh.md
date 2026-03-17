# Qwen2-Audio-finetune
这是一份为微调 Qwen2-Audio准备的仓库，支持 GPU 和 NPU，数据支持 ark 和 wav 格式的读写,支持DDP/deepspeed和lora.
# Requires
以下是运行本项目所需的依赖环境：
```
plaintext复制
numpy==1.26.0
torch==2.1.0
torch-npu==2.1.0.post10 (如果使用NPU)
torchaudio==2.1.0
torchvision==0.16.0
soundfile
transformers==4.46.3
hydra
OmegaConf
deepspeed
```
# Train
## 数据准备
请参考示例数据路径 /data/aishell-1 准备数据。
```
multitask.jsonl 
my_wav.scp
multiprompt.jsonl
```
在train.sh中修改下面两个变量，或者干脆使用默认提供的数据，不进行准备
```
TRIAN_DATA_PATH
EVAL_DATA_PATH
```
## 配置准备
在 train.sh 中设置以下必要环境变量：
```
LOCAL_DIR=
MODEL_PATH=
```
在config/config.py中设置需要的变量

## 运行代码
运行以下命令开始训练：
```
bash train.sh
```
## 解码
```
bash infer.sh
```
# RoadMap

## 注意事项
数据路径：请确保 train_data_path 和 eval_data_path 指向正确的数据目录。
设备选择：根据你的硬件环境，将 device_type 设置为 npu 或 cuda，如果不使用npu,请注释所有的import torch_npu，如果遇到和npu有关的报错，统统修改为cuda
依赖安装：请确保所有依赖已正确安装。如果未安装，可以使用以下命令：
```
pip install numpy==1.26.0 torch==2.1.0 torch-npu==2.1.0.post10 torchaudio==2.1.0 torchvision==0.16.0
```