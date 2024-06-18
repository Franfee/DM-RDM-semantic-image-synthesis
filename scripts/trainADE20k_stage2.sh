torchrun --standalone --nproc_per_node=4 train.py \
	--data="datasets/ADEChallengeData2016" --data_mode="ade20k" --resolution=256 \
	--batch=3 --batch-gpu=768 \
	--lr=1e-4 --dropout=0.13 --ls=1 --augment=0