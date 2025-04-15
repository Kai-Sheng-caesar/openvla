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
> （待完成）
> 
> 执行`cd openvla`，`pip install -e .`时报错无法连接github上的dlimp_openvla
>
> 1、添加本地代理
> 
> 2、采取手动下载https://github.com/moojink/dlimp_openvla 进行`pip install -e .`
> 
> 后续Flash Attention 2进行安装也报错
> 
> 手动下载flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
> 
> 原本要求的flash_attn_2.5.5对应cuda版本为12.2，尝试使用符合版本的2.7.3
>
> 由于无法连接到github，尝试手动安装whl进行编译

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
> 发现出现如下报错
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
torch 2.5.1 requires sympy==1.13.1, but you have sympy 1.13.3 which is incompatible.
```
> 需要更换sympy库版本为1.13.1与torch2.5.1适配，执行以下命令
```
pip install sympy==1.13.1
```

### 待完成
### 下载libero-spatial数据集

## 复现
### 使用官方openvla-7b-finetuned-libero-spatial模型进行验证
## 微调
### 在libero-spatial数据集上使用LORA微调官方openvla-7b模型
