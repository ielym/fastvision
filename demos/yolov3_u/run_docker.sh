docker run -itd \
	--runtime=nvidia \
	-e NVIDIA_VISIBLE_DEVICES=all \
	-e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
	--shm-size=512g \
	-v /home/ymluo/datasets1:/home/ymluo/datasets1 \
	-v /home/ymluo/detection/01_yolov3:/app/src \
	--name lym_detect01 \
	ielym/torch:cuda-11.4.2-cudnn8-ubuntu20.04-pytorch-1.10.1-torchvision-0.11.2 bash

