import kagglehub

# Download latest version
path = kagglehub.dataset_download("muhammadkhalid/sign-language-for-numbers")

print("Path to dataset files:", path)