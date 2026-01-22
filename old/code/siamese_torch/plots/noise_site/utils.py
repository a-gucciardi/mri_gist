import random
import numpy as np
from PIL import Image, ImageDraw
from datasets import load_dataset
from keras import ops
from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold

def add_noise(image, size='big', noise_type='noisy', rgb=False, force_shape=0):
    """
    Add a random noise area to the upper or lower half of the image, close to the middle.

    Parameters:
    - image: PIL.Image object.
    - shape: Shape of the noise area ('circle', 'square', or 'polygon').
    - size: string indicating the size of the noise area ('big', 'mid', 'lil').

    Returns:
    - image with added noise area.
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)
    shapes = ['circle', 'square']#, 'polygon']
    if force_shape == 0 or force_shape == 1:
        shape = shapes[force_shape]
    else:
        shape = shapes[random.randint(0, len(shapes) -1 )]

    #big 40x40 medium 20x20 small 10x10
    if size == 'big':
        x1, y1 = random.randint(66, 129), 105
        x2, y2 = x1+40, y1+40
    elif size == 'mid':
        x1, y1 = random.randint(66, 149), random.randint(106, 124)
        x2, y2 = x1+20, y1+20
    elif size == 'lil':
        x1, y1 = random.randint(66, 159), random.randint(106, 134)
        x2, y2 = x1+10, y1+10

    bbox = (x1, y1, x2, y2)

    if noise_type == 'plain':
        fill_color = 255        # Plain white
        if rgb: fill_color = (255, 255, 255)
    elif noise_type == 'noisy':
        # fill_color = 255
        if rgb: fill_color = tuple(random.randint(200, 215) for _ in range(3))  # Noisy white
        else: fill_color = random.randint(140, 215)

    # if bbox:
    if shape == 'circle':
        draw.ellipse(bbox, fill=fill_color)
    elif shape == 'square':
        draw.rectangle(bbox, fill=fill_color)
        # print('square')

    return image


## siamese torch 2
# TODO FIX calls


def siamese_noise_dataset_fold2(test_size=0.2, n_splits=5, shuffle=False, noise_size='big', resize=True, t1=True, t2=True, random_state=4):
    """
    Creates n folds from the data without duplicating samples and alternates between T1 and T2 image types
    """
    slice1, slice2, labels, image_types = [], [], [], []

    # T1 and T2
    if resize:
        dst1 = hf_dataset['train'].map(transforms, batched=True)
        dst2 = hf_dataset['test'].map(transforms, batched=True)
    else:
        dst1 = hf_dataset['train'].map(transforms_noresize, batched=True)
        dst2 = hf_dataset['test'].map(transforms_noresize, batched=True)

    # iterators for T1 and T2 data, used below
    t1_iter = enumerate(zip(dst1['slice1'], dst1['slice2']))
    t2_iter = enumerate(zip(dst2['slice1'], dst2['slice2']))

    # Alternate between T1 and T2 using itertools
    for (i1, (t1_slice1, t1_slice2)), (i2, (t2_slice1, t2_slice2)) in itertools.zip_longest(t1_iter, t2_iter, fillvalue=(None, (None, None))):
        # T1 data
        if t1 and i1 is not None:
            if i1 % 2 == 0:
                slice1.append(t1_slice1)
                slice2.append(t1_slice2)
                labels.append(0)
            else:
                if random.choice([True, False]):
                    slice1.append(add_noise(t1_slice1, size=noise_size))
                    slice2.append(t1_slice2)
                else:
                    slice1.append(t1_slice1)
                    slice2.append(add_noise(t1_slice2, size=noise_size))
                labels.append(1)
            image_types.append(0)  # T1 type

        # T2 data
        if t2 and i2 is not None:
            if i2 % 2 == 0:
                slice1.append(t2_slice1)
                slice2.append(t2_slice2)
                labels.append(0)
            else:
                if random.choice([True, False]):
                    slice1.append(add_noise(t2_slice1, size=noise_size))
                    slice2.append(t2_slice2)
                else:
                    slice1.append(t2_slice1)
                    slice2.append(add_noise(t2_slice2, size=noise_size))
                labels.append(1)
            image_types.append(1)  # T2 type

    ds_ = Dataset.from_dict({
        'slice1': slice1,
        'slice2': slice2,
        'label': labels,
        'image_type': image_types
    })

    print(f"Total dataset size: {len(ds_)}")
    print(f"Number of T1 images: {sum(1 for t in image_types if t == 0)}")
    print(f"Number of T2 images: {sum(1 for t in image_types if t == 1)}")

    # splits with val, stratified
    ds_trainval, ds_test = paired_stratified_split(ds_, test_size=test_size, stratify_by='image_type', random_state=random_state)

    # Prepare data for the stratified k-fold split
    indices = np.arange(len(ds_trainval) // 2)
    image_types = np.array(ds_trainval['image_type'][::2]) # type of first half of each pair

    # kfold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_indices = list(skf.split(indices, image_types))

    ds_folds = []
    for fold, (train_pair_idx, val_pair_idx) in enumerate(fold_indices):
        # Convert pair indices to image indices
        train_idx = np.concatenate([2*train_pair_idx, 2*train_pair_idx+1])
        val_idx = np.concatenate([2*val_pair_idx, 2*val_pair_idx+1])

        # Sort indices to maintain original order
        train_idx.sort()
        val_idx.sort()

        # Create train and validation splits for this fold
        ds_train = ds_trainval.select(train_idx)
        ds_val = ds_trainval.select(val_idx)

        # Add this fold to the list of folds
        ds_folds.append(DatasetDict({
            'train': ds_train,
            'valid': ds_val,
            'test': ds_test
        }))

    return ds_folds

def add_noise_range(image, size=0.5, noise_type='noisy', rgb=False, force_shape=0):
    """
    Add a random noise area to the upper or lower half of the image, close to the middle.

    Parameters:
    - image: PIL.Image object.
    - shape: Shape of the noise area ('circle', 'square', or 'polygon').
    - size: range indicating the size of the noise area (0 none to 1 maximum (50,50))

    Returns:
    - image with added noise area.
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)
    shapes = ['circle', 'square']#, 'polygon']
    if force_shape == 0 or force_shape == 1:
        shape = shapes[force_shape]
    else:
        shape = shapes[random.randint(0, len(shapes) -1 )]

    #big 40x40 medium 20x20 small 10x10
    #Xx (66)
    x1 = random.randint(66, 120/size)
    if size == 'big':
        x1, y1 = random.randint(66, 129), 105
        x2, y2 = x1+40, y1+40
    elif size == 'mid':
        x1, y1 = random.randint(66, 149), random.randint(106, 124)
        x2, y2 = x1+20, y1+20
    elif size == 'lil':
        x1, y1 = random.randint(66, 159), random.randint(106, 134)
        x2, y2 = x1+10, y1+10

    bbox = (x1, y1, x2, y2)

    if noise_type == 'plain':
        fill_color = 255        # Plain white
        if rgb: fill_color = (255, 255, 255)
    elif noise_type == 'noisy':
        # fill_color = 255
        if rgb: fill_color = tuple(random.randint(200, 215) for _ in range(3))  # Noisy white
        else: fill_color = random.randint(140, 215)

    # if bbox:
    if shape == 'circle':
        draw.ellipse(bbox, fill=fill_color)
    elif shape == 'square':
        draw.rectangle(bbox, fill=fill_color)
        # print('square')

    return image

def siamese_noise_dataset_fold_range(test_size=0.2, n_splits=5, shuffle=False, noise_size='big', resize=True, t1=True, t2=True, random_state=4):
    """
    Creates n folds from the data without duplicating samples and alternates between T1 and T2 image types
    AND range of noise size, with modified add_noise
    """
    slice1, slice2, labels, image_types = [], [], [], []

    # T1 and T2
    if resize:
        dst1 = hf_dataset['train'].map(transforms, batched=True)
        dst2 = hf_dataset['test'].map(transforms, batched=True)
    else:
        dst1 = hf_dataset['train'].map(transforms_noresize, batched=True)
        dst2 = hf_dataset['test'].map(transforms_noresize, batched=True)

    # iterators for T1 and T2 data, used below
    t1_iter = enumerate(zip(dst1['slice1'], dst1['slice2']))
    t2_iter = enumerate(zip(dst2['slice1'], dst2['slice2']))

    # Alternate between T1 and T2 using itertools
    for (i1, (t1_slice1, t1_slice2)), (i2, (t2_slice1, t2_slice2)) in itertools.zip_longest(t1_iter, t2_iter, fillvalue=(None, (None, None))):
        # T1 data
        if t1 and i1 is not None:
            if i1 % 2 == 0:
                slice1.append(t1_slice1)
                slice2.append(t1_slice2)
                labels.append(0)
            else:
                if random.choice([True, False]):
                    slice1.append(add_noise(t1_slice1, size=noise_size))
                    slice2.append(t1_slice2)
                else:
                    slice1.append(t1_slice1)
                    slice2.append(add_noise(t1_slice2, size=noise_size))
                labels.append(1)
            image_types.append(0)  # T1 type

        # T2 data
        if t2 and i2 is not None:
            if i2 % 2 == 0:
                slice1.append(t2_slice1)
                slice2.append(t2_slice2)
                labels.append(0)
            else:
                if random.choice([True, False]):
                    slice1.append(add_noise(t2_slice1, size=noise_size))
                    slice2.append(t2_slice2)
                else:
                    slice1.append(t2_slice1)
                    slice2.append(add_noise(t2_slice2, size=noise_size))
                labels.append(1)
            image_types.append(1)  # T2 type

    ds_ = Dataset.from_dict({
        'slice1': slice1,
        'slice2': slice2,
        'label': labels,
        'image_type': image_types
    })

    print(f"Total dataset size: {len(ds_)}")
    print(f"Number of T1 images: {sum(1 for t in image_types if t == 0)}")
    print(f"Number of T2 images: {sum(1 for t in image_types if t == 1)}")

    # splits with val, stratified
    ds_trainval, ds_test = paired_stratified_split(ds_, test_size=test_size, stratify_by='image_type', random_state=random_state)

    # Prepare data for the stratified k-fold split
    indices = np.arange(len(ds_trainval) // 2)
    image_types = np.array(ds_trainval['image_type'][::2]) # type of first half of each pair

    # kfold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_indices = list(skf.split(indices, image_types))

    ds_folds = []
    for fold, (train_pair_idx, val_pair_idx) in enumerate(fold_indices):
        # Convert pair indices to image indices
        train_idx = np.concatenate([2*train_pair_idx, 2*train_pair_idx+1])
        val_idx = np.concatenate([2*val_pair_idx, 2*val_pair_idx+1])

        # Sort indices to maintain original order
        train_idx.sort()
        val_idx.sort()

        # Create train and validation splits for this fold
        ds_train = ds_trainval.select(train_idx)
        ds_val = ds_trainval.select(val_idx)

        # Add this fold to the list of folds
        ds_folds.append(DatasetDict({
            'train': ds_train,
            'valid': ds_val,
            'test': ds_test
        }))

    return ds_folds