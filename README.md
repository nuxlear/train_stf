
# 설치
CUDA 11.1 기준  설치방법

```
$ sudo apt-get install ffmpeg
$ conda create --name stf python==3.7.9
$ conda activate stf
$ git clone https://github.com/ai-anchor-kr/stf_api.git
$ cd stf_api
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install -r requirements.txt

```
