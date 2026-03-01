#Channel-based structure learning framework
+ Towards efficient deep spiking neural networks construction with spiking activity based pruning
+ Framework: Spikingjelly, Pytorch
+ Dataset: CIFAR10, CIFAR100, DVS-CIFAR10

##Dependency
+ We use a framework for building spiking neural network called Spikingjelly which is based on Pytorch.
+ Install Pytorch = 1.13.1+cuda11.6:
	```
	conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
	```
+ Install Spikingjelly from GitHub:

	```
	git clone https://github.com/fangwei123456/spikingjelly.git
	cd spikingjelly
    git checkout 0.0.0.0.12
	python setup.py install
	```

##Install
+ Install the dependencies:
	```
	pip install -r requirements.txt
	```

##Organization
- mask_manage.py: Update and document changes to the network structure.
- train.py: Train based on the SCA framework. 
- snnvgg.py: The spiking VGG model.
- remove_pruned_channels.py: Completely remove the channels corresponding to zero positions in the mask to obtain the compressed network model.
- utils.py: Training-related utility functions.

##Running Examples
+ Train based on the SCA framework. 
	#Train a lightweight SNN model with 30% of the original number of channels.
    ```
	python train.py --alpha 0.1 --beta 0.2 --lr 0.01
    ```
	