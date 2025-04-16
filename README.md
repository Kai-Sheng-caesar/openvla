# OpenVLA复现及LIBERO数据集上微调
## 配置环境及准备工作

### 克隆OpenVLA仓库
```
# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch. Below is a sample command to do this, but you should check the following link
# to find installation instructions that are specific to your compute platform:
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  # UPDATE ME!

# Clone and install the openvla repo
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```
> 对于无法连接到github有以下方法
> 
> 执行`cd openvla`，`pip install -e .`时报错无法连接github上的dlimp_openvla
>
> 1、添加本地代理
> 
> 2、手动下载https://github.com/moojink/dlimp_openvla 进行`pip install -e .`
> 
> 后续Flash Attention 2进行安装也报错
> 
> 手动下载flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
> 
> 原本要求的flash_attn_2.5.5对应cuda版本为12.2，尝试使用符合版本的2.7.3
>
> 由于无法连接到github，尝试手动安装whl进行编译
>
> 3、编辑`pyproject.toml`或者`setup.py`中的下载链接，采用国内镜像站kkgithub、hf.mirror等国内镜像源进行`pip install -e .`
>
> 采用第2种方法完成了flash_attn_2.7.3的安装
>
> 
> 采用第3种方法完成了openvla中的`pip install -e .`
#### 后续...
> flash_attn_2.7.3在测试运行`test.py`测试代码报错，原作者说一定要使用他的版本！确实没说错，复现一定要尽可能使用相同的环境！
>
> 后续将flash_attn_2.7.3换成了作者要求的flash_attn_2.5.5

### 克隆LIBERO仓库
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

### 安装其余LEBERO相关库
```
cd openvla
pip install -r experiments/robot/libero/libero_requirements.txt
```
> 出现如下报错：
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
torch 2.5.1 requires sympy==1.13.1, but you have sympy 1.13.3 which is incompatible.
```
> 需要更换sympy库版本为1.13.1与torch2.5.1适配，执行以下命令
```
pip install sympy==1.13.1
```

## 测试基础的openvla-7b模型，运行官方示例test.py
> `test.py`:
```
# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

# Load Processor & VLA
processor = AutoProcessor.from_pretrained('./openvla7b', trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    './openvla7b',
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

# Grab image input & format prompt


image: Image.Image = Image.open("./test.png")
# prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"
prompt = "In: What action should the robot take to {<put eggplant in bowl>}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(action)

# Execute...
# robot.act(action, ...)
```
> 完善示例代码中的图片读取，测试图片为：
<img src="test.png" width="30%">

> 完善示例代码中的prompt，文本提示为：
>
> `prompt = "In: What action should the robot take to {<put eggplant in bowl>}?\nOut:"`
>
> 结果为：
<img src="result.png" width="80%">

## 待完成
### 下载libero-spatial（空间位置）数据集用以后续复现及微调工作

## 进一步复现
### 使用官方针对上述数据集微调的openvla-7b-finetuned-libero-spatial模型进行验证

## 自己尝试微调
### 在libero-spatial数据集上自己尝试使用LORA微调官方openvla-7b模型
