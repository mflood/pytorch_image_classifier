
#python predict.py assets/japanese.jpg checkpoints/checkpoint_densenet121_ft500_ep3.pth --top_k 5 --category_names cat_to_name.json
#python predict.py assets/30.jpg checkpoints/checkpoint_densenet121_ft500_ep3.pth --top_k 5 --category_names cat_to_name.json
#python predict.py assets/azalea.jpg.jpg checkpoints/checkpoint_densenet121_ft500_ep3.pth --top_k 5 --category_names cat_to_name.json


checkpoints="checkpoint_densenet121_ft400_ep1.pth checkpoint_densenet121_ft500_ep3.pth checkpoint_densenet121_ft512_ep3.pth checkpoint_vgg13_ft300_ep3.pth checkpoint_vgg16_ft600_ep4.pth"
assets="japanese.jpg 30.jpg azalea.jpg"

for checkpoint in $checkpoints; do
    for image in $assets; do
	python predict.py assets/$image checkpoints/$checkpoint --top_k 5 --category_names cat_to_name.json
    done
done
