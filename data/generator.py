from __future__ import print_function
import torch.utils.data as data
from PIL import Image as pil_image
import torch
import numpy as np
import random
from torch.autograd import Variable
from . import omniglot
from . import mini_imagenet
from . import tiered_imagenet


class Generator(data.Dataset):
    def __init__(self, root, args, partition='train', dataset='omniglot'):
        self.root = root
        self.partition = partition  # training set or test set
        self.args = args

        assert (dataset == 'omniglot' or
                dataset == 'mini_imagenet' or
                dataset == 'tiered_imagenet'
                ), 'Incorrect dataset partition'
        self.dataset = dataset

        if self.dataset == 'omniglot':
            self.input_channels = 1
            self.size = (28, 28)
        elif self.dataset == 'mini_imagenet':
            self.input_channels = 3
            self.size = (84, 84)
        elif self.dataset == 'tiered_imagenet':
            self.input_channels = 3
            self.size = (84, 84)
        else:
            raise ValueError('unknown dataset')

        if dataset == 'omniglot':
            self.loader = omniglot.Omniglot(self.root, dataset=dataset)
            self.data = self.loader.load_dataset(self.partition == 'train', self.size)
        elif dataset == 'mini_imagenet':
            self.loader = mini_imagenet.MiniImagenet(self.root)
            self.data, self.label_encoder = self.loader.load_dataset(self.partition, self.size)
        elif dataset == 'tiered_imagenet':
            self.loader = tiered_imagenet.TieredImagenet(self.root)
            self.data = self.loader.load_dataset(self.partition, self.size)
        else:
            raise NotImplementedError

        self.class_encoder = {}
        for id_key, key in enumerate(self.data):
            self.class_encoder[key] = id_key

    def rotate_image(self, image, times):
        rotated_image = np.zeros(image.shape)
        for channel in range(image.shape[0]):
            rotated_image[channel, :, :] = np.rot90(image[channel, :, :], k=times)
        return rotated_image

    def load_image_samples(self, samples):
        if self.dataset != 'tiered_imagenet':
            return samples

        samples_data = []
        for sample in samples:
            image_data = np.array(pil_image.open(sample), dtype='float32')

            # WARNING! BGR -> RGB
            image_data = image_data[..., ::-1]

            # Normalize
            image_data = np.transpose(image_data, (2, 0, 1))
            image_data[0, :, :] -= 120.45  # R
            image_data[1, :, :] -= 115.74  # G
            image_data[2, :, :] -= 104.65  # B
            image_data /= 127.5

            samples_data.append(image_data)
        return samples_data

    def get_task_batch(self, batch_size=5, n_way=20, num_shots=1, transductive=False, unlabeled_extra=0, cuda=False, variable=False):
        # Init variables
        # xs stands for list consists of multiple queries (num_queries == n_way)
        batch_xs, labels_xs, labels_xs_global = [], [], []

        # should consider remove single query initialization
        # batch_x = np.zeros((batch_size, self.input_channels, self.size[0], self.size[1]), dtype='float32')
        # labels_x = np.zeros((batch_size, n_way), dtype='float32')
        # labels_x_global = np.zeros(batch_size, dtype='int64')

        target_distances = np.zeros((batch_size, n_way * num_shots), dtype='float32')
        hidden_labels = np.zeros((batch_size, n_way * num_shots + n_way), dtype='float32')

        numeric_labels = []
        batches_xi, labels_yi, oracles_yi = [], [], []

        for i in range(n_way * num_shots):
            batches_xi.append(np.zeros((batch_size, self.input_channels, self.size[0], self.size[1]), dtype='float32'))
            labels_yi.append(np.zeros((batch_size, n_way), dtype='float32'))
            oracles_yi.append(np.zeros((batch_size, n_way), dtype='float32'))

        # both case return list (len(list) == 1 vs n_ways)
        if transductive:
            for i in range(n_way):
                batch_xs.append(np.zeros((batch_size, self.input_channels, self.size[0], self.size[1]), dtype='float32'))
                labels_xs.append(np.zeros((batch_size, n_way), dtype='float32'))
                labels_xs_global.append(np.zeros(batch_size, dtype='int64'))
        else:
            batch_xs.append(np.zeros((batch_size, self.input_channels, self.size[0], self.size[1]), dtype='float32'))
            labels_xs.append(np.zeros((batch_size, n_way), dtype='float32'))
            labels_xs_global.append(np.zeros(batch_size, dtype='int64'))

        # Iterate over tasks for the same batch
        for batch_counter in range(batch_size):
            # class selected for query (not applicable for transductive)
            positive_class = random.randint(0, n_way - 1) # [0, n_way-1]

            # Sample random classes for this TASK
            classes_ = list(self.data.keys())
            sampled_classes = random.sample(classes_, n_way)
            indexes_perm = np.random.permutation(n_way * num_shots)

            counter = 0
            for class_counter, class_ in enumerate(sampled_classes):
                if transductive:
                    # every class take num_shots + 1
                    samples = random.sample(self.data[class_], num_shots + 1)
                    samples = self.load_image_samples(samples)
                    # take first sample for all classes
                    batch_xs[class_counter][batch_counter, :, :, :] = samples[0]
                    labels_xs[class_counter][batch_counter, class_counter] = 1
                    labels_xs_global[class_counter][batch_counter] = self.class_encoder[class_]
                    samples = samples[1::]
                elif class_counter == positive_class:  # non-transductive
                    # We take num_shots + 1 samples for one class
                    samples = random.sample(self.data[class_], num_shots + 1)
                    samples = self.load_image_samples(samples)
                    # Test sample is loaded (1st sample)
                    batch_xs[0][batch_counter, :, :, :] = samples[0]
                    labels_xs[0][batch_counter, class_counter] = 1  # one-hot
                    labels_xs_global[0][batch_counter] = self.class_encoder[class_]
                    samples = samples[1::]
                else:
                    samples = random.sample(self.data[class_], num_shots)
                    samples = self.load_image_samples(samples)

                for s_i in range(0, len(samples)):
                    batches_xi[indexes_perm[counter]][batch_counter, :, :, :] = samples[s_i]
                    if s_i < unlabeled_extra:
                        labels_yi[indexes_perm[counter]][batch_counter, class_counter] = 0
                        hidden_labels[batch_counter, indexes_perm[counter] + 1] = 1
                    else:
                        labels_yi[indexes_perm[counter]][batch_counter, class_counter] = 1
                    oracles_yi[indexes_perm[counter]][batch_counter, class_counter] = 1
                    target_distances[batch_counter, indexes_perm[counter]] = 0
                    counter += 1

            numeric_labels.append(positive_class)  # this is not returned

        batch_xs = [torch.from_numpy(batch_x) for batch_x in batch_xs]
        labels_xs = [torch.from_numpy(labels_x) for labels_x in labels_xs]
        labels_xs_global = [torch.from_numpy(labels_x_global) for labels_x_global in labels_xs_global]

        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]
        oracles_yi = [torch.from_numpy(oracle_yi) for oracle_yi in oracles_yi]

        labels_xs_scalar = [np.argmax(labels_x, 1) for labels_x in labels_xs]  # discrete

        # return_arr = [torch.from_numpy(batch_x), torch.from_numpy(labels_x), torch.from_numpy(labels_x_scalar),
        #               torch.from_numpy(labels_x_global), batches_xi, labels_yi, oracles_yi,
        #               torch.from_numpy(hidden_labels)]
        return_arr = [batch_xs, labels_xs, labels_xs_scalar, labels_xs_global,
                      batches_xi, labels_yi, oracles_yi,
                      torch.from_numpy(hidden_labels)]

        if cuda:
            return_arr = self.cast_cuda(return_arr)
        if variable:
            return_arr = self.cast_variable(return_arr)
        return return_arr

    def cast_cuda(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda()
        return input

    def cast_variable(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_variable(input[i])
        else:
            return Variable(input)

        return input
