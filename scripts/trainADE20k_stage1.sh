torchrun --standalone --nproc_per_node=4 train.py \
	--data="datasets/ADEChallengeData2016" --data_mode="ade20k" --resolution=64 \
	--batch=64 --batch-gpu=1024 \
	--lr=1e-4 --dropout=0.15 \
