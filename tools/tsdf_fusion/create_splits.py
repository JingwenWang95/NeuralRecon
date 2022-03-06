import os

DATA_ROOT = "/media/jingwen/Data/scannet/scans"
TEST_ROOT = "/media/jingwen/Data/scannet/scans_test"
split_prefix = "scannetv2_"


def save_splits(splits, save_dir):
    for split in splits:
        with open(os.path.join(save_dir, split_prefix + "{}.txt".format(split)), "w") as f:
            for scene in splits[split]:
                f.write(scene + "\n")


# get 100 scenes
trainval = sorted(os.listdir(DATA_ROOT))[:100]
test = sorted(os.listdir(TEST_ROOT))[:20]
splits = dict(train=trainval[:13] + trainval[18:], val=trainval[13:18], test=test)
save_splits(splits, "/media/jingwen/Data/neuralrecon/scannet/100_tsdf_9/splits")