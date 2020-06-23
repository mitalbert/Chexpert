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
        
        frequently = lambda aug: iaa.Sometimes(0.9, aug)
        often = lambda aug: iaa.Sometimes(0.7, aug)
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        occasionally = lambda aug: iaa.Sometimes(0.3, aug)
        rarely = lambda aug: iaa.Sometimes(0.1, aug)
        
        if aug_level == 'high':
            self.aug = iaa.Sequential([
                
                sometimes(iaa.Crop(percent=(0, 0.1))),
                rarely(iaa.ContrastNormalization((0.75, 1.5))),
                occasionally(iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.1, 0.2))),

                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-20, 20),
                    shear=(-8, 8),
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL
                )),
                
                iaa.SomeOf((0, 5),
                [

                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),

                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                # Search in some images either for all edges or for
                # directed edges. These edges are then marked in a black
                # and white image and overlayed with the original image
                # using an alpha of 0 to 0.7.
                sometimes(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),

                # Add gaussian noise to some images.
                # In 50% of these cases, the noise is randomly sampled per
                # channel and pixel.
                # In the other 50% of all cases it is sampled once per
                # pixel (i.e. brightness change).
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.03*255), per_channel=0.5
                ),

                # Either drop randomly 1 to 10% of all pixels (i.e. set
                # them to black) or drop them on an image with 2-5% percent
                # of the original size, leading to large dropped
                # rectangles.
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.01, 0.02),
                        per_channel=0.2
                    ),
                ]),

                # Invert each image's channel with 5% probability.
                # This sets each pixel value v to 255-v.
                iaa.Invert(0.05, per_channel=True), # invert color channels

                # Add a value of -10 to 10 to each pixel.
                iaa.Add((-10, 10), per_channel=0.5),

                # Change brightness of images (50-150% of original value).
                iaa.Multiply((0.5, 1.0), per_channel=0.5),

                # Improve or worsen the contrast of images.
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

                # In some images move pixels locally around (with random
                # strengths).
                sometimes(
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                ),

                # In some images distort local areas with varying strength.
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
            ], random_order=True)

            ], random_order=True)
            
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
        img = PIL.Image.fromarray(np.uint8(img)) # convert to pil image
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
