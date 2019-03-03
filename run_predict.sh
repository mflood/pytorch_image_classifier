python predict.py assets/30.jpg ic_checkpoint.pth  -v --top_k 7 
python predict.py assets/30.jpg ic_checkpoint.pth  -v --top_k 1
python predict.py assets/30.jpg ic_checkpoint.pth  -v --top_k 5 --category_names cat_to_name.json
python predict.py assets/azalea.jpg ic_checkpoint.pth --top_k 15 --category_names cat_to_name.json
python predict.py assets/japanese.jpg ic_checkpoint.pth  --top_k 5 --category_names cat_to_name.json
