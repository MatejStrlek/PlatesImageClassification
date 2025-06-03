from torch.utils.data import Dataset
from PIL import Image

class LicensePlateDataset(Dataset):
    def __init__(self, dataframe, image_size=(224, 128), transform=None):
        self.dataframe = dataframe
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepaths']
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size)

        if self.transform:
            image = self.transform(image)

        label = self.dataframe.iloc[idx]['label_encoded']
        return image, label