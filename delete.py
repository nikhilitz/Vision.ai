import os

# Specify the directory path
folder_path = '/Users/nikhilgupta/Desktop/Deep Learning/Vision.ai/Data/images/Flicker8k_Dataset/'

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Filter out the specific images (e.g., images with '.1' in the filename)
files_to_delete = [f for f in files if '.1' in f]

# Loop through the files and delete them
for file in files_to_delete:
    file_path = os.path.join(folder_path, file)
    try:
        os.remove(file_path)
        print(f"Deleted: {file}")
    except Exception as e:
        print(f"Error deleting {file}: {e}")
