import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image as pil_image


class TieredImagenet(data.Dataset):
    def __init__(self, root='./datasets', dataset='tiered_imagenet'):
        self.root = root
        self.dataset = dataset
        assert self._check_exists_()

    def _check_exists_(self):
        if not os.path.exists(os.path.join(
                self.root, 'tiered-imagenet')) or not os.path.exists(
                    os.path.join(self.root, 'tiered-imagenet')):
            return False
        else:
            return True

    def load_dataset(self, partition, size=(84, 84)):
        print('Loading tiered-imagenet dataset')
        labels_name = '{}/tiered-imagenet/{}_labels.pkl'.format(self.root, partition)
        images_dir = '{}/tiered-imagenet/{}_images'.format(self.root, partition)
        assert os.path.exists(images_dir)

        # load images and labels
        try:
            with open(labels_name, 'rb') as f:
                labels = pickle.load(f, encoding='bytes')['label_specific']
            print('read label data: {}'.format(labels.shape))
        except:
            raise IOError('labels_name file cannot read properly')

        data = {}
        n_classes = np.max(labels) + 1
        counter = 0
        for c_idx in range(n_classes):
            data[c_idx] = []
            cnt = len(np.where(labels == c_idx)[0])
            for i in range(cnt):
                img_name = os.path.join(images_dir, '{}_{}_{}.png'.format(c_idx, i, counter))
                assert os.path.exists(img_name), 'missing image of {}'.format(img_name)
                data[c_idx].append(img_name)
                counter += 1

                # image2resize = pil_image.fromarray(np.uint8(images[i, ...]))
                # image_resized = image2resize.resize((size[1], size[0]))
                # image_resized = np.array(image_resized, dtype='float32')

                # # Normalize
                # image_resized = np.transpose(image_resized, (2, 0, 1))
                # image_resized[0, :, :] -= 120.45  # R
                # image_resized[1, :, :] -= 115.74  # G
                # image_resized[2, :, :] -= 104.65  # B
                # image_resized /= 127.5

                # data[c_idx].append(image_resized)

        return data


if __name__ == "__main__":
    ds = TieredImagenet()
