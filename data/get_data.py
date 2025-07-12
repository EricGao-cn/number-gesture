import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("muhammadkhalid/sign-language-for-numbers")

os.system(f"mv {path} ./data")
