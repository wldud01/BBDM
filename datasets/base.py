from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

# 이미지 변환 및 전처리 관련
class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), flip=False, to_normal=False):
        self.image_size = image_size    
        self.image_paths = image_paths  
        self._length = len(image_paths) 
        self.flip = flip    
        self.to_normal = to_normal  

    def __len__(self):
        if self.flip:   
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        # 증강을 적용하는 경우, 절반에 해당하는 인덱스는 좌우 반전 이미지로 처리
        if index >= self._length:   
            index = index - self._length
            p = 1.0

        # 이미지 전처리
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),   
            transforms.Resize(self.image_size), 
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        # CT 이미지이므로, grayscale로 변환
        if image.mode == 'RGB':
            image = image.convert('L') # default = 'RGB'


        image = transform(image)

        # -1~1로 정규화
        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        image_name = Path(img_path).stem

        
        return image, image_name
