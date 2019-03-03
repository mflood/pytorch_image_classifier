#python train.py flowers -v --arch=vgg13
python train.py flowers --arch=densenet121 --checkpoint=checkpoints/checkpoint_densenet121_ft500_ep3.pth --save_dir=checkpoints
python train.py flowers --arch=densenet121 --save_dir=checkpoints --epochs=1 --hidden_units=400 --save_dir=checkpoints
python train.py flowers --arch=vgg13 --save_dir=checkpoints --epochs=3 --hidden_units=300 --save_dir=checkpoints
python train.py flowers --arch=vgg16 --save_dir=checkpoints --epochs=4 --hidden_units=600 --save_dir=checkpoints
