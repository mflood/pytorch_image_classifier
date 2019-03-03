python predict.py assets/30.jpg checkpoints/ic_checkpoint.pth --top_k 5 --category_names cat_to_name.json
python predict.py assets/azalea.jpg checkpoints/ic_checkpoint.pth --top_k 15 --category_names cat_to_name.json
python predict.py assets/japanese.jpg checkpoints/ic_checkpoint.pth --top_k 5 --category_names cat_to_name.json
python predict.py assets/japanese.jpg checkpoints/checkpoint_densenet121_ft500_ep3.pth --top_k 5 --category_names cat_to_name.json
