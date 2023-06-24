import os
import zipfile
from PIL import Image
import shutil

# Dataset name
DATASET_NAME = "PetImages"


class Dataset:
    # Image file extensions
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']

    def __init__(self, dataset_path):
        # Set dataset path
        self.dataset_path = dataset_path

    def is_image_file(self, filename):
        """Check if file is an image file with extensions defined in IMG_EXTENSIONS."""
        return any(filename.lower().endswith(extension) for extension in self.IMG_EXTENSIONS)

    def delete_file(self, img_path):
        """Delete file if it's an invalid image file."""
        if os.path.isdir(img_path):
            # Remove directory if it is empty
            os.rmdir(img_path)
        else:
            # Remove file if it exists and print message
            os.remove(img_path)
            print(f"Deleted invalid image file: {img_path}")

    def is_jpg(self, img_path):
        """Check if file is a JPEG file."""
        try:
            # Open image file
            i = Image.open(img_path)
            return i.format == 'JPEG'
        except IOError:
            return False

    def filter_dataset(self):
        """Filter dataset by deleting invalid image files."""
        for sub_dir in os.listdir(self.dataset_path):
            for file_name in os.listdir(os.path.join(self.dataset_path, sub_dir)):
                img_path = os.path.join(os.path.join(self.dataset_path, sub_dir, file_name))
                if not (self.is_image_file(file_name) and self.is_jpg(img_path)):
                    # Delete invalid image file
                    self.delete_file(img_path)
                    continue

    def make_dirs(self, path):
        """Create directory and its parent directories if they don't exist."""
        os.makedirs(path, exist_ok=True)

    def join_path(self, path, *paths):
        """Join paths."""
        return os.path.join(path, *paths)

    def split_dataset(self, eval_split=0.1):
        """Split dataset into train and evaluation sets."""
        self.make_dirs(self.join_path(self.dataset_path, "train"))
        self.make_dirs(self.join_path(self.dataset_path, "eval"))
        for sub_dir in os.listdir(self.dataset_path):
            if sub_dir in ["train", "eval"]:
                continue
            cls_list = os.listdir(self.join_path(self.dataset_path, sub_dir))
            train_size = int(len(cls_list) * (1 - eval_split))
            self.make_dirs(self.join_path(self.dataset_path, "train", sub_dir))
            self.make_dirs(self.join_path(self.dataset_path, "eval", sub_dir))
            for i, file_name in enumerate(os.listdir(self.join_path(self.dataset_path, sub_dir))):
                source_file = self.join_path(self.dataset_path, sub_dir, file_name)
                if i <= train_size:
                    target_file = self.join_path(self.dataset_path, "train", sub_dir, file_name)
                else:
                    target_file = self.join_path(self.dataset_path, "eval", sub_dir, file_name)
                shutil.move(source_file, target_file)
            # Delete directory after files have been moved
            self.delete_file(self.join_path(self.dataset_path, sub_dir))

    @staticmethod
    def extract_dataset(zip_file, save_dir):
        """Extract dataset from zip file to save directory."""
        if not os.path.exists(zip_file):
            raise ValueError(f"Zip file {zip_file} does not exist!")
        try:
            print(f"Extracting dataset from {zip_file} to {save_dir}")
            zip_file = zipfile.ZipFile(zip_file)
            for names in zip_file.namelist():
                zip_file.extract(names, save_dir)
            zip_file.close()
            print(f"Successfully extracted dataset at {Dataset.join_path(save_dir, DATASET_NAME)}")
        except:
            raise ValueError(f"Failed to extract dataset from {zip_file}!")


# Main function
if __name__ == '__main__':
    # Set save directory path
    save_dir = os.path.abspath("./dataset")
    # Create dataset directory if it doesn't exist
    Dataset(save_dir).make_dirs(save_dir)
    zip_file = "cat_dog.zip"
    # Extract dataset from zip file to save directory
    Dataset.extract_dataset(zip_file, save_dir)
    dataset = Dataset(os.path.join(save_dir, DATASET_NAME))
    # Filter dataset by deleting invalid image files
    dataset.filter_dataset()
    # Split dataset into train and evaluation sets
    dataset.split_dataset(eval_split=0.1)
