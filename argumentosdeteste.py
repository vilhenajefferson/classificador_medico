import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--k_value", type=int, help="LIDAR test data file, if you want to merge multiple"
                                                                   " datasets, simply provide a list of paths, as follows:"
                                                                   " --lidar_training_data path_a.npz path_b.npz")
parser.add_argument("--folder", type=str, help="Path, where the model is saved")

args = parser.parse_args()

print("args.folder = ", args.folder)
print("args.k_value = ", args.k_value)