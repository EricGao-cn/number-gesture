import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("muhammadkhalid/sign-language-for-numbers")

os.system(f"mv {path} ./data")
# os.system("mkdir dataset")
# os.system("mv 1/*/* dataset")
# os.system("rm -rf dataset/unknown")

# print("Path to dataset files:", path)