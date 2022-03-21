import os

DATA_ROOT = "/media/jingwen/Data/scannet/scans"
TEST_ROOT = "/media/jingwen/Data/scannet/scans_test"
split_prefix = "scannetv2_"


def save_splits(splits, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for split in splits:
        with open(os.path.join(save_dir, split_prefix + "{}.txt".format(split)), "w") as f:
            for scene in splits[split]:
                f.write(scene + "\n")


# get all scenes
trainval = sorted(os.listdir(DATA_ROOT))
test = sorted(os.listdir(TEST_ROOT))
splits = dict(train=trainval[:-50], val=trainval[-50:], test=test)
save_splits(splits, "/media/jingwen/Data/neuralrecon/scannet/all_tsdf_9/splits")