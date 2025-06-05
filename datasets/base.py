from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

# 이미지 변환 및 전처리 관련
class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), flip=False, to_normal=False):
        self.image_size = image_size    # 이미지 크기
        self.image_paths = image_paths  # 이미지 경로
        self._length = len(image_paths) # 
        self.flip = flip    # 데이터 증강 여부
        self.to_normal = to_normal  # -1~1로 정규화 여부

    def __len__(self):
        if self.flip:   # 증강하는 경우, 이미지 수를 2배로 간주하여 augmentaton 적용
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
            transforms.RandomHorizontalFlip(p=p),   #  좌우 반전
            transforms.Resize(self.image_size), # 이미지 크기 조정
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
