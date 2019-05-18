import os
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage import io, transform
from time import sleep

import torch
from torchvision import transforms, utils
from torch.utils.data.dataset import Dataset

class ImgSeqDataset(Dataset):
    def __init__(self, directory, n_sequences, max_seq_length, transform = None):
        sequences = os.listdir(directory)
        data = []
        if n_sequences is None:
            n_sequences = len(sequences)

        for i, seq in enumerate(sequences):

            # regulate the number of sequences
            if i >= n_sequences:
                break

            # extract data from the correct directory
            dir = os.path.join(directory, seq)
            data.append(self.process_sequence(dir))

        # break up data into sequences of length < max_seq_length
        processed_data = []
        for seq in data:
            for i in range(0, len(seq), max_seq_length):
                if i != 0:
                    processed_data.append(seq[i - max_seq_length: i])

        # store processed data
        self.data = processed_data

        # store transform
        self.transform = transform

    # method to get check if acceptable transform
    @classmethod
    def is_supported(cls, transform):
        return isinstance(transform, MyTransforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # reset random transforms
        if self.transform is not None:
            for tr in self.transform.transforms:
                randomize = getattr(tr, "randomize", None)

                if callable(randomize):
                    randomize()

        # get appropriate sequence
        seq = self.data[idx]

        # process sequence
        processed = self.load_sequence(seq)

        return processed



    def load_sequence(self, sequence):
        p_images = []
        p_landmarks = []
        p_labels = []
        processed = []

        for info in sequence:

            if len(info) == 0:
                print("No path found for image")
                continue

            # get path to image
            img_path = info[0]

            # get label information
            annotations = info[1:]

            # read image
            img = io.imread(img_path)

            # get image dims
            height, width, depth = img.shape

            # get points, labels
            landmarks = []
            labels = []
            for i in range(len(annotations)):

                if i % 5 == 0:
                    x = int(float(annotations[i]) * width)
                if i % 5 == 1:
                    y = int(float(annotations[i]) * height)
                if i % 5 == 2:
                    x2 = int(float(annotations[i]) * width)
                if i % 5 == 3:
                    y2 = int(float(annotations[i]) * height)
                if i % 5 == 4:
                    point1 = np.asarray([x, y])
                    point2 = np.asarray([x2, y2])
                    label = int(float(annotations[i]))

                    landmarks.append(point1)
                    landmarks.append(point2)
                    labels.append(label)

            # convert to numpy
            landmarks = np.asarray(landmarks).astype('float').reshape(-1, 2)
            labels = np.asarray(labels).astype('long')

            # add results to processed sets
            p_images.append(img)
            p_landmarks.append(landmarks)
            p_labels.append(labels)

        # create sample
        sample = {'images': np.asarray(p_images),
                  'landmarks': np.asarray(p_landmarks),
                  'labels': np.asarray(p_labels)}

        # apply transform if necessary
        if self.transform is not None:
            sample = self.transform(sample)

        # extract processed info
        p_images = sample['images']
        p_landmarks = sample['landmarks']
        p_labels = sample['labels']

        return p_images, p_landmarks, p_labels




    def display_image_label(self, image, landmarks, labels):

        # making sure inputs are numpy arrays
        image = np.asarray(image).transpose(1, 2, 0)
        landmarks = np.asarray(landmarks)
        labels = np.asarray(labels)

        # set up colors
        colors = ['r', 'g', 'b', 'y', 'o', 'p']

        # Create figure and axes
        fig, ax = plt.subplots(1)


        height, width, depth = image.shape

        # visualize the image
        ax.imshow(image, animated=True)
        boxes = []

        # add rectangles to image
        for i in range(len(landmarks)):

            if i%4 == 0:
                x, y = landmarks[i]
            if i%4 == 1:
                x2, y2 = landmarks[i]

                # Create a Rectangle patch
                rect = patches.Rectangle((x, y), x2-x, y2-y, linewidth=1, edgecolor=colors[labels[i//2]], facecolor='none')
                ax.add_patch(rect)
                boxes.append(rect)

        plt.show()
        # plt.show(block=False)
        # plt.pause(1/3)
        #
        # # remove boxes
        # for box in boxes:
        #     box.remove()
        #
        # plt.close()



    def process_sequence(self, path):

        # create empty list for storing sequence data
        data = []

        # try to find annotation .csv file
        annotation_file = self.find_annotation_csv(path)

        # check that annotation is not None
        if annotation_file is None:
            return data

        # extract data from csv file
        with open(annotation_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                # parse row as necessary
                for col in range(len(row)):

                    # check that elem isnt empty
                    if row[col] != '':
                        # update path to current location on disk
                        if col == 0:
                            p_cur = row[col]
                            file = os.path.basename(p_cur)
                            p_new = os.path.join(path, file)
                            row[col] = p_new
                    # if elem empty remove end of list and finish iterating
                    else:
                        row = row[:col]
                        break

                data.append(row)

        return data


    # function for finding csv annotation file within a specified directory
    def find_annotation_csv(self, directory):
        # setting up variable
        annotation_file = None

        # look for results directory
        res_dir = None
        for dir in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, dir)) and ("Results" in dir or "results" in dir):
                res_dir = os.path.join(directory, dir)
                break

        # check that results directory was found
        if res_dir is None:
            return None

        # look for csv file in directory that contains key word Final or final
        for file in os.listdir(res_dir):
            if ("Final" in file or "final" in file) and file.endswith(".csv"):
                annotation_file = os.path.join(res_dir, file)
                break

        # check that annotation file was found
        if annotation_file is not None:
            return annotation_file
        else:
            print("WARNING: no annotation file found at {}".format(dir))
            return None

class MyTransforms:
    pass

class Rescale(MyTransforms, object):

    def __init__(self, output_size):
        assert len(output_size) == 2
        self.output_size = output_size

    def __call__(self, sample):
        images, landmarks = sample['images'], sample['landmarks']

        # determining old dimensions
        h, w = images.shape[1:3]

        # determine new image dimensions
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        new_shape = (images.shape[0], new_h, new_w, images.shape[3])

        # applying resize transform
        imgs = transform.resize(images, new_shape)

        # h and w are swapped for landmarks because for images, x and y axes are axis 1 and 0 respectively
        for i, landmark in enumerate(landmarks):
            landmarks[i] = landmark * [new_w / w, new_h / h]

        # update dicts
        sample['images'] = imgs
        sample['landmarks'] = landmarks

        return sample

class RandomCrop(MyTransforms, object):

    def __init__(self, input_size, output_size):
        assert len(output_size) == 2
        assert len(input_size) == 2
        self.output_size = output_size
        self.input_size = input_size

        # determine crop dims
        h, w = self.input_size
        new_h, new_w = self.output_size
        self.top = np.random.randint(0, h - new_h)
        self.left = np.random.randint(0, w - new_w)

    def __call__(self, sample):
        images, landmarks = sample['images'], sample['landmarks']

        # determine and check old image dimensions
        h, w = images.shape[1:3]
        assert h == self.input_size[0] and w == self.input_size[1]

        # get output image dims
        new_h, new_w = self.output_size

        # crop image
        images = images[:, self.top: self.top + new_h,  self.left: self.left + new_w]

        # update landmarks
        for i, landmark in enumerate(landmarks):
            landmarks[i] = landmark - [self.left, self.top]

        # update dicts
        sample['images'] = images
        sample['landmarks'] = landmarks

        return sample

    def randomize(self):
        # determine crop dims
        h, w = self.input_size
        new_h, new_w = self.output_size
        self.top = np.random.randint(0, h - new_h)
        self.left = np.random.randint(0, w - new_w)


class ToTensor(MyTransforms, object):

    def __call__(self, sample):
        images = sample['images']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        images = images.transpose((0, 3, 1, 2))

        # update dicts
        sample['images'] = torch.from_numpy(images).type(torch.float)

        return sample

class Normalize(MyTransforms, object):

    def __call__(self, sample):
        images = sample['images']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        images = images/255

        # update dicts
        sample['images'] = torch.from_numpy(images).type(torch.float)

        return sample

def collate_fn(batch):

    images = [item[0] for item in batch]
    landmarks = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    # converting to torch
    images = np.asarray([t.numpy() for t in images])

    # images = self.add_label_image(images, landmarks)

    images = torch.tensor(images, dtype=torch.float)

    return [images, landmarks, labels]
#
# A = ImgSeqDataset('D:\\Anton\\data\\hand_gesture_sequences\\test', 3, 5)
# for i in range(len(A)):
#     processed = A[i]
#
#     imgs, landmark, label =processed
#
#     for j in range(imgs.shape[0]):
#         A.display_image_label(imgs[j].transpose(2, 0, 1), landmark[j], label[j])

#     for sample in processed:
#
#         print(i, sample['image'].shape, sample['landmarks'].shape, len(sample['labels']))
#
#         A.display_image_label(**sample)
#
#         if i == 3:
#             plt.show()
#             break