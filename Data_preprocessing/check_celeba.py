from training.image_datasets import load_data


dataset_iterator = load_data(
        dataset_mode="celeba",
        data_dir="/root/autodl-tmp/CelebA-HQ",
        batch_size=1,
        image_size=64,
        class_cond=True,
        is_train=True
    )

for i in range(30000):
    try:
        images, cond = next(dataset_iterator)
    except Exception as E:
        print(E)
        print("err in read: ",i)
        continue

    if i % 100 ==0 :
        print(f"check {i}")    