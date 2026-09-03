import kagglehub

# Download latest version
path = kagglehub.dataset_download("orvile/wesad-wearable-stress-affect-detection-dataset")

print("Path to dataset files:", path)