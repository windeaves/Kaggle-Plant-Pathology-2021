import random
from torchvision.transforms import functional as F

class Transformation:
    "Data augmentation using torchvision.transfoms.functional"

    def __init__(self, rotate_range=(-30, 30), brightness_range=(0.7, 1.5), contrast_range=(0.7, 1.5),
                 scale_range=(0.7, 1.5), shear_range=(-30, 30), tanslate_range=(-30, 30)):
        self.threshold = 0.5
        self.angle = 0
        self.brightness_factor = 1
        self.contrast_factor = 1
        self.scale = 1
        self.shear = 0
        self.translate = (0, 0)
        self.size = 512
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.rotate_range = rotate_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.tanslate_range = tanslate_range

    def rotate(self, x):
        return F.rotate(x, self.angle)

    def adjustBrightness(self, x):
        return F.adjust_brightness(x, self.brightness_factor)

    def adjustContrast(self, x):
        return F.adjust_contrast(x, self.contrast_factor)

    def affine(self, x):
        return F.affine(x, 0, self.translate, self.scale, self.shear)
    
    def vflip(self, x):
        return F.vflip(x)

    def hflip(self, x):
        return F.hflip(x)

    def normalize(self, x):
        return F.normalize(x, self.mean, self.std)

    def to_tensor(self, x):
        return F.to_tensor(x)

    def resize(self, x):
        return F.resize(x, (self.size, self.size))

    def geo_transform(self, image):
        # Random Rotation
        if random.random() > self.threshold:
            self.angle = random.randint(*self.rotate_range)
            image = self.rotate(image)

        # Random vertical flip
        if random.random() > self.threshold:
            image = self.vflip(image)

        # Random horizontal flip
        if random.random() > self.threshold:
            image = self.hflip(image)

        # Random Affine
        if random.random() > self.threshold:
            self.scale = random.uniform(*self.scale_range)
            self.shear = random.uniform(*self.shear_range)
            self.translate = (random.uniform(*self.tanslate_range), random.uniform(*self.tanslate_range))
            image = self.affine(image)

        return image
    
    def color_transform(self, image):
        # Random Adjust Brightness
        if random.random() > self.threshold:
            self.brightness_factor = random.uniform(*self.brightness_range)
            image = self.adjustBrightness(image)

        # Random Adjust Contrast
        if random.random() > self.threshold:
            self.contrast_factor = random.uniform(*self.contrast_range)
            image = self.adjustContrast(image)

        return image

    def __call__(self, image):
        image = self.geo_transform(image)
        image = self.color_transform(image)
        
        image = self.resize(image)
        image = self.to_tensor(image)
        image = self.normalize(image)

        return image