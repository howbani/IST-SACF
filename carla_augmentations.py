import torch
from kornia import augmentation as K

class IdentityAugmentation:
    def __init__(self, input_shape):
        assert len(input_shape) == 2, "Input shape must be 2D"
        self.input_shape = input_shape
        self.output_shape = input_shape

    def evaluation_augmentation(self, image):
        return image

    def training_augmentation(self, image_batch):
        return image_batch

    def _reshape_to_frame_batch(self, image_batch):
        if image_batch.shape[1] % 3 != 0:
            raise ValueError("Expected channel dimension to be divisible by 3.")
        frame_stack = image_batch.shape[1] // 3
        frame_batch = image_batch.reshape(-1, 3, *self.input_shape)
        return frame_batch, frame_stack

    def _reshape_from_frame_batch(self, frame_batch, frame_stack):
        return frame_batch.reshape(
            -1,
            3 * frame_stack,
            self.input_shape[0],
            self.input_shape[1],
        )


class ColorJiggle(IdentityAugmentation):
    def __init__(self, input_shape):
        super().__init__(input_shape)
        self.aug = K.ColorJiggle(brightness=0.0,
                                 contrast=0.2,
                                 saturation=0.5,
                                 hue=0.1,
                                 same_on_batch=False,
                                 p=0.85,
                                 keepdim=True)

    def training_augmentation(self, image_batch):
        image_batch = image_batch / 255.0
        frame_batch, frame_stack = self._reshape_to_frame_batch(image_batch)
        frame_batch = self.aug(frame_batch)
        image_batch = self._reshape_from_frame_batch(frame_batch, frame_stack)
        image_batch = image_batch * 255.0
        return torch.clamp(image_batch, 0, 255)


class GaussianNoise(IdentityAugmentation):
    def __init__(self, input_shape):
        super().__init__(input_shape)
        self.aug = K.RandomGaussianNoise(mean=0.0, std=10.0, p=1.0)

    def training_augmentation(self, image_batch):
        frame_batch, frame_stack = self._reshape_to_frame_batch(image_batch)
        frame_batch = self.aug(frame_batch)
        image_batch = self._reshape_from_frame_batch(frame_batch, frame_stack)
        return torch.clamp(image_batch, 0, 255)


class ColorJiggleAndGaussianNoise(IdentityAugmentation):
    def __init__(self, input_shape):
        super().__init__(input_shape)
        self.color_aug = K.ColorJiggle(brightness=0.0,
                                       contrast=0.2,
                                       saturation=0.5,
                                       hue=0.1,
                                       same_on_batch=False,
                                       p=0.85,
                                       keepdim=True)
        self.noise_aug = K.RandomGaussianNoise(mean=0.0, std=10.0, p=1.0)

    def training_augmentation(self, image_batch):
        image_batch = image_batch / 255.0
        frame_batch, frame_stack = self._reshape_to_frame_batch(image_batch)
        frame_batch = self.color_aug(frame_batch)
        image_batch = self._reshape_from_frame_batch(frame_batch, frame_stack)
        image_batch = image_batch * 255.0

        frame_batch, frame_stack = self._reshape_to_frame_batch(image_batch)
        frame_batch = self.noise_aug(frame_batch)
        image_batch = self._reshape_from_frame_batch(frame_batch, frame_stack)
        return torch.clamp(image_batch, 0, 255)


def make_augmentor(name, input_shape):
    print(f'CHOSEN AUGMENTATION: {name}')
    augmentor = None
    if name == 'color_jiggle':
        augmentor = ColorJiggle(input_shape)
    elif name == 'gaussian_noise':
        augmentor = GaussianNoise(input_shape)
    elif name == 'color_jiggle_and_gaussian_noise':
        augmentor = ColorJiggleAndGaussianNoise(input_shape)
    else:
        raise ValueError('augmentation is not supported: %s' % name)
    return augmentor
