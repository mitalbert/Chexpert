import cv2
import torchvision.transforms as tfs
import imgaug

# for imgaug
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import PIL

class ImgAugTransform:
    def __init__(self,aug_level):
        if aug_level == 'high':
            self.aug = iaa.Sequential([
                iaa.Sometimes(0.99,
                              iaa.GaussianBlur(sigma=(0, 1.0)),
                              iaa.ContrastNormalization((0.75, 1.5)),
                              # iaa.OneOf([iaa.Multiply((0.1, 0.5)),iaa.MultiplyElementwise((0.1, 0.5))])
                             ),
                
                iaa.Sometimes(0.8, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=0.5)),
                
                iaa.Sometimes(0.2,
                              iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                         iaa.CoarseDropout((0.0, 0.1), size_percent=(0.01, 0.2))])),
    #             iaa.PiecewiseAffine(scale=(0.01, 0.05)), # takes lots of time
    #             iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True), # only for colored images
                iaa.Sometimes(0.5, iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-20, 20),
                    shear=(-8, 8)
                ))
            ], random_order=False)

        elif aug_level == 'low':
            self.aug = iaa.Sequential([
                iaa.OneOf([
                iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0.0, 0.1*255))),
                iaa.Sometimes(0.1, iaa.AdditiveGaussianNoise(scale=(0.0, 0.3*255)))
                          ]),
                
                iaa.Sometimes(0.5, iaa.GammaContrast((0.5, 2.0))),
                iaa.Sometimes(0.5, iaa.imgcorruptlike.Brightness(severity=2)),
                iaa.Sometimes(0.5, iaa.Affine(
                    rotate=(-20, 20),
                )),
            ], random_order=True)
        else:
            raise NotImplementedError
      
    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
#         img = np.ascontiguousarray(res) # this fixes "some of the strides of a given numpy array are negative"
        img = PIL.Image.fromarray(np.uint8(img))
        return img

def Common(image):

    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


def Aug(image):
    img_aug = tfs.Compose([
        tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05),
                         scale=(0.95, 1.05), fillcolor=128)
    ])
    image = img_aug(image)

    return image

def GetTransforms(image, target=None, type='common'):
    # taget is not support now
    if target is not None:
        raise Exception(
            'Target is not support now ! ')
    # get type
    if type.strip() == 'Common':
        image = Common(image)
        return image
    elif type.strip() == 'None':
        return image
    elif type.strip() == 'Aug':
        image = Aug(image)
        return image

    elif type.strip() == 'imgaug_Low':
        imgaugment = ImgAugTransform('low')
        
        compose = tfs.Compose([
        tfs.RandomApply([imgaugment], p=0.7),
        tfs.RandomHorizontalFlip(),
#         normalize,
        ])
        
        image = compose(image)

        return image
        

    elif type.strip() == 'imgaug_High':
        # look at their normalization
        # maybe replace their data loader with std torch one

        imgaugment = ImgAugTransform('high')

        compose = tfs.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.Resize((35,20)),
        tfs.RandomApply([imgaugment], p=0.7),
        tfs.RandomHorizontalFlip(),
#         normalize,
        ])
        
        image = compose(image)

        return image

    else:
        raise Exception(
            'Unknown transforms_type : '.format(type))
