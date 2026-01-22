import cv2, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from datasets import load_dataset
from keras import ops
from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold
import itertools

hf_dataset = load_dataset("agucci/mri-sym2")

def non_zeros(img, plot = True):
    """
        Detects the coordinates of all non-zero (non-black) pixels along the four edges of the image. 
        These points are returned into a single list of (x, y).

        img : PIL Image
    """

    # img = example['half1_noise']
    img_array = np.asarray(img.convert("L"))
    height, width = img_array.shape

    # Find the edge points from the top line
    edge_points = []
    min_x, min_y = width, height
    max_x, max_y = 0, 0
    for y in range(height):
        for x in range(width):
            if img_array[y, x]:
                edge_points.append((x, y))
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    if plot: 
        # Plot the half1_noise and edge points
        plt.figure(figsize=(4, 4))
        plt.imshow(img_array, cmap='gray')
        if edge_points:
            x_points, y_points = zip(*edge_points)
            plt.scatter(x_points, y_points, color='red', s=10)  # Plot edge points as red dots
        plt.title('Brain segmentation - non zeros pixels')
        plt.axis('off')
        plt.show()
        # Print the number of intersection points for reference
        print("Number of Edge Points:", len(edge_points))
        print(f"Min coordinates: ({min_x}, {min_y})")
        print(f"Max coordinates: ({max_x}, {max_y})")

    return edge_points

def segment2(img, edge_points):
    """
        Segment the the brain part from the black background.

        img : PIL Image
    """

    img_array = np.array(img.convert('L'))
    height, width = img_array.shape

    # Initialize a binary mask
    mask = np.zeros_like(img_array, dtype=np.uint8)

    for x, y in edge_points[:width]:
        mask[y, x] = 255

    for x, y in edge_points[width:2*width]:
        mask[y, x] = 255

    for x, y in edge_points[2*width:2*width+height]:
        mask[y, x] = 255

    for x, y in edge_points[2*width+height:]:
        mask[y, x] = 255

    rgba_img = np.zeros((height, width, 4), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            rgba_img[y, x, :3] = img_array[y, x]  # Copy intensity from original half1_noise
            rgba_img[y, x, 3] = mask[y, x]       # Set transparency from mask

    transparent_img = Image.fromarray(rgba_img, 'RGBA')

    return transparent_img

def rotate(img, line, show_line = False):
    """
        img : PIL Image
    """

    img_array = np.array(img)
    height, width = img_array.shape[:2]
    # rgba_img = np.zeros((height, width, 4), dtype=np.uint8)

    # Draw the (long) line
    draw = ImageDraw.Draw(img.copy())

    line_coords = eval(line.replace('} {', '}, {'))

    for i in range(len(line_coords) - 1):
        start_point = (line_coords[i]['x'], line_coords[i]['y'])
        end_point = (line_coords[i+1]['x'], line_coords[i+1]['y'])

        print(start_point, end_point)
        # slope & intercept + border points
        a = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
        b = start_point[1] - a * start_point[0]
        left_point = (0, int(b))                # (0, b)
        right_point = (width, int(width*a+b))   # (290, 290*a + b)

        angle = np.arctan2(right_point[1] - left_point[1],
                right_point[0] - left_point[0]) * 180 / np.pi
        center_x = (left_point[0] + right_point[0]) / 2
        center_y = (left_point[1] + right_point[1]) / 2
        # 
        center_new_x = img_array.shape[1] / 2
        center_new_y = img_array.shape[0] / 2

        # print(f"Line points : {left_point}, {right_point} \nAngle : {angle}")
        # print((center_x, center_y))
        # print((center_half1_noise_x, center_half1_noise_y))

        # Translation
        offset_x = center_new_x - center_x
        offset_y = center_new_y - center_y
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        translated_img = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]))

        rotation_matrix = cv2.getRotationMatrix2D((center_new_x, center_new_y), angle, 1)
        rotated_img = cv2.warpAffine(translated_img, rotation_matrix, (img_array.shape[1], img_array.shape[0]))

        rotated_img_pil = Image.fromarray(rotated_img, 'RGBA')
        
        # (Might not be necessary)
        # Calculate the offset caused by the rotation
        rotated_center = np.dot(rotation_matrix, np.array([center_x, center_y, 1]))
        offset_x_rotated = center_new_x - rotated_center[0]
        offset_y_rotated = center_new_y - rotated_center[1]
        # Adjust the position of the rotated half1_noise
        M_adjusted = np.float32([[1, 0, offset_x_rotated], [0, 1, offset_y_rotated]])
        final_img = cv2.warpAffine(rotated_img, M_adjusted, (img_array.shape[1], img_array.shape[0]))
        final_img_pil = Image.fromarray(final_img, 'RGBA')

        # Draw the (long) line
        if show_line:
            draw1, draw2 = ImageDraw.Draw(rotated_img_pil), ImageDraw.Draw(final_img_pil)
            draw1.line([left_point, right_point], fill=(255, 0, 0, 255), width=1)
            draw2.line([(0, 145.0), (290, 145.0)], fill=(255, 0, 0, 255), width=1)

    return final_img_pil

def slice_aligned(transparent_img):
    """
        Slice but for when the Image is aligned to the ((0, 145) (290, 145)) line
    """

    # intercept = 145
    _, height = transparent_img.size
    intercept = int(height / 2)

    img_array = np.array(transparent_img)#.convert('L'))
    reshaped_1 = Image.fromarray(img_array[intercept:, :, :], 'RGBA')
    reshaped_2 = Image.fromarray(img_array[:intercept, :, :], 'RGBA')

    return(reshaped_1.convert("L"), reshaped_2.convert("L"))

def draw_line(img, line):

    img = img.copy()
    draw = ImageDraw.Draw(img)

    line_coords = eval(line.replace('} {', '}, {'))

    for i in range(len(line_coords) - 1):
        start_point = (line_coords[i]['x'], line_coords[i]['y'])
        end_point = (line_coords[i+1]['x'], line_coords[i+1]['y'])
        draw.line([start_point, end_point], fill=(255, 0, 0, 255), width=4)

    return(img)

def segment(img, line, intersection_points, show_line = False, rotate_img = False):
    """
        Segment the the brain part from the black background, and draw the annotation line. 

        img : PIL Image
    """

    img_array = np.array(img.convert('L'))
    height, width = img_array.shape

    # Initialize a binary mask
    mask = np.zeros_like(img_array, dtype=np.uint8)

    for x, y in intersection_points[:width]:
        mask[y, x] = 255

    for x, y in intersection_points[width:2*width]:
        mask[y, x] = 255

    for x, y in intersection_points[2*width:2*width+height]:
        mask[y, x] = 255

    for x, y in intersection_points[2*width+height:]:
        mask[y, x] = 255

    rgba_img = np.zeros((height, width, 4), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            rgba_img[y, x, :3] = img_array[y, x]  # Copy intensity from original half1_noise
            rgba_img[y, x, 3] = mask[y, x]       # Set transparency from mask

    transparent_img = Image.fromarray(rgba_img, 'RGBA')

    # Draw the (long) line
    draw = ImageDraw.Draw(transparent_img.copy())

    line_coords = eval(line.replace('} {', '}, {'))

    for i in range(len(line_coords) - 1):
        start_point = (line_coords[i]['x'], line_coords[i]['y'])
        end_point = (line_coords[i+1]['x'], line_coords[i+1]['y'])

        # slope & intercept + border points
        a = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
        b = start_point[1] - a * start_point[0]
        left_point = (0, int(b))                # (0, b)
        right_point = (width, int(width*a+b))   # (290, 290*a + b)

        if rotate_img:
            angle = np.arctan2(right_point[1] - left_point[1],
                    right_point[0] - left_point[0]) * 180 / np.pi
            center_x = (left_point[0] + right_point[0]) / 2
            center_y = (left_point[1] + right_point[1]) / 2
            center_half1_noise_x = rgba_img.shape[1] / 2
            center_half1_noise_y = rgba_img.shape[0] / 2

            # print(f"Line points : {left_point}, {right_point} \nAngle : {angle}")
            # print((center_x, center_y))
            # print((center_half1_noise_x, center_half1_noise_y))

            # Translation
            offset_x = center_half1_noise_x - center_x
            offset_y = center_half1_noise_y - center_y
            M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
            translated_half1_noise = cv2.warpAffine(rgba_img, M, (rgba_img.shape[1], rgba_img.shape[0]))

            rotation_matrix = cv2.getRotationMatrix2D((center_half1_noise_x, center_half1_noise_y), angle, 1)
            rotated_half1_noise = cv2.warpAffine(translated_half1_noise, rotation_matrix, (rgba_img.shape[1], rgba_img.shape[0]))

            #print(type(rotated_half1_noise), rotated_half1_noise.shape)
            rotated_img = Image.fromarray(rotated_half1_noise, 'RGBA')

            # Calculate the offset caused by the rotation
            rotated_center = np.dot(rotation_matrix, np.array([center_x, center_y, 1]))
            offset_x_rotated = center_half1_noise_x - rotated_center[0]
            offset_y_rotated = center_half1_noise_y - rotated_center[1]
            # Adjust the position of the rotated half1_noise
            M_adjusted = np.float32([[1, 0, offset_x_rotated], [0, 1, offset_y_rotated]])
            final_half1_noise = cv2.warpAffine(rotated_half1_noise, M_adjusted, (rgba_img.shape[1], rgba_img.shape[0]))
            final_img = Image.fromarray(final_half1_noise, 'RGBA')

        # Draw the (long) line
        if show_line:
            draw1, draw2, draw3 = ImageDraw.Draw(transparent_img), ImageDraw.Draw(rotated_img), ImageDraw.Draw(final_img)
            draw1.line([left_point, right_point], fill=(255, 0, 0, 255), width=1)
            draw2.line([left_point, right_point], fill=(255, 0, 0, 255), width=1)
            draw3.line([(0, 145.0), (290, 145.0)], fill=(255, 0, 0, 255), width=1)
        # else : draw = ImageDraw.Draw(transparent_img.copy())


    return transparent_img, rotated_img, final_img

def translate_image(image, dx, dy):
    """
    Translate the image by dx pixels horizontally and dy pixels vertically.
    """
    # PIL to np
    img_array = np.array(image)

    translated_array = np.roll(img_array, (dx, dy), axis=(1, 0))

    # np to PIL
    translated_image = Image.fromarray(translated_array)

    return translated_image

def resize_image(image, new_size=(256, 256)):
    """
    Resize the image to the new_size and fill the added part with empty RGBA.
    """
    # PIL 256, 256
    new_image = Image.new("RGBA", new_size, (0, 0, 0, 255))

    # Paste the original image onto the new image, centered
    x_offset = (new_size[0] - image.size[0]) // 2
    y_offset = (new_size[1] - image.size[1]) // 2
    # print("x &y offs", x_offset, y_offset)
    new_image.paste(image, (x_offset, y_offset))

    return new_image

def slice(transparent_img, line, test = False):
    """
    TODEL
    """

    width, height = transparent_img.size
    resize = (256, 256)

    line_coords = eval(line.replace('} {', '}, {'))
    print("line", line_coords)

    slope = (line_coords[1]['y'] - line_coords[0]['y']) / (line_coords[1]['x'] - line_coords[0]['x'])
    intercept = line_coords[0]['y'] - slope * line_coords[0]['x']

    if test:
        slope = 0
        intercept = height/2 #145.0

    # Initialize two empty half1_noises for the two halves
    half1_img = Image.new('RGBA', transparent_img.size, (255, 255, 255, 0))
    half2_img = Image.new('RGBA', transparent_img.size, (255, 255, 255, 0))

    # Iterate over each pixel in the half1_noise and separate into two halves based on symmetry line
    for y in range(height):
        for x in range(width):
            # Calculate the expected y-coordinate for the current x-coordinate based on the symmetry line
            expected_y = slope * x + intercept

            if 0 <= x <= width:
                if y >= expected_y:
                    # Above the symmetry line
                    half1_img.putpixel((x, y), transparent_img.getpixel((x, y)))
                else:
                    # Below the symmetry line
                    half2_img.putpixel((x, y), transparent_img.getpixel((x, y)))

    half1_img = resize_image(half1_img, resize)
    half2_img = resize_image(half2_img, resize)
    half1_img = translate_image(half1_img, 0, -64 )   # move vertically 1/4 of 256
    half2_img = translate_image(half2_img, 0,  64 )

    return(half1_img, half2_img)

def slice_adjusted(transparent_img):
    """
        Slice but for when the Image is aligned to the ((0, 145) (290, 145)) line
    """

    resize = (256, 256)
    # intercept = 145
    width, height = transparent_img.size
    intercept = int(height / 2)

    img_array = np.array(transparent_img)#.convert('L'))
    reshaped_1 = Image.fromarray(img_array[intercept:, :, :], 'RGBA')
    reshaped_2 = Image.fromarray(img_array[:intercept, :, :], 'RGBA')

    reshaped_1 = resize_image(reshaped_1, resize)
    reshaped_2 = resize_image(reshaped_2, resize)
    # reshaped_translated_1 = translate_image(reshaped_1, 0, -64 )   # 290: 64, 217: 54
    # reshaped_translated_2 = translate_image(reshaped_2, 0,  64 )

    # return(reshaped_translated_1, reshaped_translated_2)
    return(reshaped_1, reshaped_2)

def extract_slices(img, line):
    intersection_points = mask(img, plot = False)
    segmented = segment(img, line, intersection_points)
    half1, half2 = slice(segmented, line)

    # return segmented
    return half1, half2

def extract_slices_adj(img, line):
    intersection_points = mask(img, plot = False)
    _, _, segmented_final = segment(img, line, intersection_points, rotate_img=True)
    half1, half2 = slice_adjusted(segmented_final)

    # return segmented
    return half1, half2

def extract_slices_adj2(img, line, noise = False):
    intersection_points = mask(img, plot = False)
    _, _, segmented = segment(img, line, intersection_points, rotate_img=True)
    if noise: segmented = add_noise(segmented)
    half1, half2 = slice_adjusted(segmented)

    # return segmented
    return half1, half2

def noise(half1, half2):
    width, height = half1.size
    half1_noise = np.array(half1)

    # Define the region of interest (ROI)
    roi_top_left = (int(height/2) -10, 70)       # Example values, adjust as needed
    roi_bottom_right = (int(height/2) +10, 90)  

    # Extract the ROI
    roi = half1_noise[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

    # Generate Gaussian noise with the same dimensions as the ROI
    mean = 0
    std_dev = 25  # Adjust as needed
    noise = np.random.normal(mean, std_dev, roi.shape).astype(np.uint8)

    # Add the generated noise to the ROI
    noisy_roi = cv2.add(roi, noise)

    # Ensure pixel values stay within the valid range
    noisy_roi = np.clip(noisy_roi, 0, 255)

    # Replace the original region with the noisy region in the half1_noise
    half1_noise[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]] = noisy_roi

    # print(half1_noise)
    # print(type(half1_noise))

    # return Image.fromarray(half1_noise), Image.fromarray(half2)
    return half1_noise, half2

def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
    """Creates a plot of pairs and labels, and prediction if it's test dataset.

    Arguments:
        pairs: Numpy Array, of pairs to visualize, having shape
               (Number of pairs, 2, 28, 28).
        to_show: Int, number of examples to visualize (default is 6)
                `to_show` must be an integral multiple of `num_col`.
                 Otherwise it will be trimmed if it is greater than num_col,
                 and incremented if if it is less then num_col.
        num_col: Int, number of images in one row - (default is 3)
                 For test and train respectively, it should not exceed 3 and 7.
        predictions: Numpy Array of predictions with shape (to_show, 1) -
                     (default is None)
                     Must be passed when test=True.
        test: Boolean telling whether the dataset being visualized is
              train dataset or test dataset - (default False).

    Returns:
        None.
    """

    # Define num_row
    # If to_show % num_col != 0
    #    trim to_show,
    #       to trim to_show limit num_row to the point where
    #       to_show % num_col == 0
    #
    # If to_show//num_col == 0
    #    then it means num_col is greater then to_show
    #    increment to_show
    #       to increment to_show set num_row to 1
    num_row = to_show // num_col if to_show // num_col != 0 else 1

    # `to_show` must be an integral multiple of `num_col`
    #  we found num_row and we have num_col
    #  to increment or decrement to_show
    #  to make it integral multiple of `num_col`
    #  simply set it equal to num_row * num_col
    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(10, 10))
    for i in range(to_show):
        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(ops.concatenate([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()

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

def cut_align(image, line, resize=True, reshape = (224,224), show=False):
    """
    Given an image and line annotation from the dataset, 
    remove background, segment, rotate and cut the two halves.
    Returns halves from the same angle.
    """
    edge_points = non_zeros(image, plot=False)
    transparent_img = segment2(image, edge_points)
    rotated_img = rotate(transparent_img, line, show_line=False)
    slice1, slice2 = slice_aligned(rotated_img)
    slice1 = slice1.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    square1, square2 = Image.new('L', (290, 290), 0), Image.new('L', (290, 290), 0) # center slice in black square
    square1.paste(slice1, (0, 50)), square2.paste(slice2, (0, 50))
    # print(square1)
    if resize:
        square1, square2 = square1.resize(reshape), square2.resize(reshape)

    # if show:
    #     display(slice1) # bottom half
    #     display(slice2) # top

    return square1, square2

def transforms(examples):
    """
    Convert to grayscale, and map cut_align to hf dataset, add column for each half
    """
    # careful with the conversion here, resize migth be needed on other datasets
    list_slices = [cut_align(im.convert("L"), line, resize=True) for im, line in zip(examples["image"], examples["line"])]
    examples["slice1"], examples["slice2"] = [i[0] for i in list_slices], [i[1] for i in list_slices]
    # examples["image_convert"] = [image.convert("L").resize((290, 290)) for image in examples["image"]]

    return examples

def transforms_noresize(examples):
    """
    transforms without resizing, temptative
    """
    # careful with the conversion here, resize migth be needed on other datasets
    list_slices = [cut_align(im.convert("L"), line, resize=False) for im, line in zip(examples["image"], examples["line"])]
    examples["slice1"], examples["slice2"] = [i[0] for i in list_slices], [i[1] for i in list_slices]
    # examples["image_convert"] = [image.convert("L").resize((290, 290)) for image in examples["image"]]

    return examples

def paired_stratified_split(dataset, test_size, stratify_by, random_state):
    """
    Used for the skfold dataset, used to replace hf train_test which didnt have stratify
    """
    # each consecutive pair has to be kept together
    pair_count = len(dataset) // 2
    pair_indices = np.arange(pair_count)
    pair_types = np.array(dataset[stratify_by][::2])  # type of first half of each pair

    # Split pair indices
    skf = StratifiedKFold(n_splits=int(1/test_size), shuffle=True, random_state=random_state)
    train_pair_indices, test_pair_indices = next(skf.split(pair_indices, pair_types))

    # Convert pair indices to image indices
    train_indices = np.concatenate([2*train_pair_indices, 2*train_pair_indices+1])
    test_indices = np.concatenate([2*test_pair_indices, 2*test_pair_indices+1])

    # Sort indices to maintain original order
    train_indices.sort()
    test_indices.sort()

    return dataset.select(train_indices), dataset.select(test_indices)

def siamese_noise_dataset(test_size=0.2, shuffle=False, noise_size='big', resize=True, t1=True, t2=True):
    """
    T1 and T2
    using the huggingface API for pytorch port
    """
    noise_slice1_t1, noise_slice1_t2, labels_t1, labels_t2 = [], [], [], []
    # T1 
    if resize:
        dst1 = hf_dataset['train'].map(transforms, batched=True)
        dst2 = hf_dataset['test'].map(transforms, batched=True)
    else:
        dst1 = hf_dataset['train'].map(transforms_noresize, batched=True)
        dst2 = hf_dataset['test'].map(transforms_noresize, batched=True)

    # noise every other sample
    # left side only 
    if t1:
        noise_slice1_t1 = [add_noise(imt1, size=noise_size) if i%2 else imt1 for i, imt1 in enumerate(dst1['slice1'])]
        labels_t1 = [1 if i%2 else 0 for i in range(len(dst1['slice1']))]
    if t2:
        noise_slice1_t2 = [add_noise(imt2, size=noise_size) if i%2 else imt2 for i, imt2 in enumerate(dst2['slice1'])]
        labels_t2 = [1 if i%2 else 0 for i in range(len(dst2['slice1']))]

    if t1 and t2: 
        ds_ = Dataset.from_dict({
            'slice1': noise_slice1_t1 + noise_slice1_t2,
            'slice2': dst1['slice2'] + dst2['slice2'],
            'label': labels_t1 + labels_t2
        }) 

    elif t1:
        ds_ = Dataset.from_dict({
            'slice1': noise_slice1_t1,
            'slice2': dst1['slice2'],
            'label': labels_t1
        })

    elif t2:
        ds_ = Dataset.from_dict({
            'slice1': noise_slice1_t2,
            'slice2': dst2['slice2'],
            'label': labels_t2
        })

    # splits with val
    ds_traintestval = ds_.train_test_split(test_size=test_size, shuffle = shuffle)
    ds_testval = ds_traintestval['test'].train_test_split(test_size=0.5, shuffle = False)

    ds_final = DatasetDict({
        'train': ds_traintestval['train'],
        'test': ds_testval['test'],
        'valid': ds_testval['train']})
    
    return ds_final

def siamese_noise_dataset_new(test_size=0.2, shuffle=False, noise_size='big', resize=True, t1=True, t2=True):
    """
    T1 and T2
    using the huggingface API for pytorch port
    changes : now the noise can also be on the second half
    """
    noise_slice1_t1, noise_slice2_t1, noise_slice1_t2, noise_slice2_t2, labels_t1, labels_t2 = [], [], [], [], [], []
    # T1 
    if resize:
        dst1 = hf_dataset['train'].map(transforms, batched=True)
        dst2 = hf_dataset['test'].map(transforms, batched=True)
    else:
        dst1 = hf_dataset['train'].map(transforms_noresize, batched=True)
        dst2 = hf_dataset['test'].map(transforms_noresize, batched=True)

    # noise randomly on one hemisphere for every other sample
    if t1:
        for i, (imt1_slice1, imt1_slice2) in enumerate(zip(dst1['slice1'], dst1['slice2'])):
            if i % 2 == 0:
                noise_slice1_t1.append(imt1_slice1)
                noise_slice2_t1.append(imt1_slice2)
                labels_t1.append(0)
            else:
                if random.choice([True, False]):
                    noise_slice1_t1.append(add_noise(imt1_slice1, size=noise_size))
                    noise_slice2_t1.append(imt1_slice2)
                else:
                    noise_slice1_t1.append(imt1_slice1)
                    noise_slice2_t1.append(add_noise(imt1_slice2, size=noise_size))
                labels_t1.append(1)

    if t2:
        for i, (imt2_slice1, imt2_slice2) in enumerate(zip(dst2['slice1'], dst2['slice2'])):
            if i % 2 == 0:
                noise_slice1_t2.append(imt2_slice1)
                noise_slice2_t2.append(imt2_slice2)
                labels_t2.append(0)
            else:
                if random.choice([True, False]):
                    noise_slice1_t2.append(add_noise(imt2_slice1, size=noise_size))
                    noise_slice2_t2.append(imt2_slice2)
                else:
                    noise_slice1_t2.append(imt2_slice1)
                    noise_slice2_t2.append(add_noise(imt2_slice2, size=noise_size))
                labels_t2.append(1)

    if t1 and t2: 
        ds_ = Dataset.from_dict({
            'slice1': noise_slice1_t1 + noise_slice1_t2,
            'slice2': noise_slice2_t1 + noise_slice2_t2,
            'label': labels_t1 + labels_t2
        }) 

    elif t1:
        ds_ = Dataset.from_dict({
            'slice1': noise_slice1_t1,
            'slice2': noise_slice2_t1,
            'label': labels_t1
        })

    elif t2:
        ds_ = Dataset.from_dict({
            'slice1': noise_slice1_t2,
            'slice2': noise_slice2_t2,
            'label': labels_t2
        })

    # splits with val
    ds_traintestval = ds_.train_test_split(test_size=test_size, shuffle=shuffle)
    ds_testval = ds_traintestval['test'].train_test_split(test_size=0.5, shuffle=False)

    ds_final = DatasetDict({
        'train': ds_traintestval['train'],
        'test': ds_testval['test'],
        'valid': ds_testval['train']})
    
    return ds_final

def siamese_noise_dataset_any(dataset):
    """
    T1 and T2
    using the huggingface API for pytorch port
    changes : now the noise can also be on the second half
    """
    noise_slice1, noise_slice2 = [], []
    
    dst1 = dataset.map(transforms, batched=True)

    # noise randomly on one hemisphere for every other sample
    for i, (imt1_slice1, imt1_slice2) in enumerate(zip(dst1['slice1'], dst1['slice2'])):
        noise_slice1.append(imt1_slice1)
        noise_slice2.append(imt1_slice2)

    ds_ = Dataset.from_dict({
        'slice1': noise_slice1,
        'slice2': noise_slice2
    })
    
    ds_final = DatasetDict({'test': ds_})
    
    return ds_final

def siamese_noise_dataset_fold(test_size=0.2, n_splits=5, shuffle=False, noise_size='big', resize=True, t1=True, t2=True, random_state=4):
    """
    similar to the previous version, but outputs a list of n folds from the data
    """
    noise_slice1_t1, noise_slice2_t1, noise_slice1_t2, noise_slice2_t2, labels_t1, labels_t2 = [], [], [], [], [], []
    # T1 
    if resize:
        dst1 = hf_dataset['train'].map(transforms, batched=True)
        dst2 = hf_dataset['test'].map(transforms, batched=True)
    else:
        dst1 = hf_dataset['train'].map(transforms_noresize, batched=True)
        dst2 = hf_dataset['test'].map(transforms_noresize, batched=True)

    # noise randomly on one hemisphere for every other sample
    if t1:
        for i, (imt1_slice1, imt1_slice2) in enumerate(zip(dst1['slice1'], dst1['slice2'])):
            if i % 2 == 0:
                noise_slice1_t1.append(imt1_slice1)
                noise_slice2_t1.append(imt1_slice2)
                labels_t1.append(0)
            else:
                if random.choice([True, False]):
                    noise_slice1_t1.append(add_noise(imt1_slice1, size=noise_size))
                    noise_slice2_t1.append(imt1_slice2)
                else:
                    noise_slice1_t1.append(imt1_slice1)
                    noise_slice2_t1.append(add_noise(imt1_slice2, size=noise_size))
                labels_t1.append(1)

    if t2:
        for i, (imt2_slice1, imt2_slice2) in enumerate(zip(dst2['slice1'], dst2['slice2'])):
            if i % 2 == 0:
                noise_slice1_t2.append(imt2_slice1)
                noise_slice2_t2.append(imt2_slice2)
                labels_t2.append(0)
            else:
                if random.choice([True, False]):
                    noise_slice1_t2.append(add_noise(imt2_slice1, size=noise_size))
                    noise_slice2_t2.append(imt2_slice2)
                else:
                    noise_slice1_t2.append(imt2_slice1)
                    noise_slice2_t2.append(add_noise(imt2_slice2, size=noise_size))
                labels_t2.append(1)

    if t1 and t2: 
        ds_ = Dataset.from_dict({
            'slice1': noise_slice1_t1 + noise_slice1_t2,
            'slice2': noise_slice2_t1 + noise_slice2_t2,
            'label': labels_t1 + labels_t2
        }) 

    elif t1:
        ds_ = Dataset.from_dict({
            'slice1': noise_slice1_t1,
            'slice2': noise_slice2_t1,
            'label': labels_t1
        })

    elif t2:
        ds_ = Dataset.from_dict({
            'slice1': noise_slice1_t2,
            'slice2': noise_slice2_t2,
            'label': labels_t2
        })

    # Create a new column to indicate image type T1(1476) or T2 
    ds_ = ds_.map(lambda example, idx: {'image_type': 0 if idx < 1476 else 1}, with_indices=True)
    # splits with val, stratified
    ds_trainval, ds_test = paired_stratified_split(ds_, test_size=test_size, stratify_by='image_type', random_state=random_state)
    print(ds_trainval)
    print(ds_test)

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

def add_noise_range(image, size=0, noise_type='noisy', rgb=False, force_shape=None):
    """
    Add a random noise area to the upper or lower half of the image, close to the middle.

    Parameters:
    - image: PIL.Image object.
    - shape: Shape of the noise area ('circle', 'square', or 'polygon').
    - size: range indicating the size of the noise area (0 (2,2) to 9 maximum (40,40))

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

    sizes = [int(2 + (40 - 2) * i / 9) for i in range(10)]  # 10 sizes from 2 to 40
    size_drawn = sizes[size]
    #big 40x40 medium 20x20 small 10x10
    #Xx (66)
    # Calculate valid x range based on size
    x1 = random.randint(61, 169 - size_drawn)
    y1 = random.randint(105, 145 - size_drawn)
    bbox = (x1, y1, x1 + size_drawn, y1 + size_drawn)

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

def add_noise_range_290(image, size=0, noise_type='noisy', rgb=False, force_shape=None):
    """
    Same as previous, but the first was buggy on the size when using 290 images. Pb with the x1 y1 definiton.
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)
    shapes = ['circle', 'square']#, 'polygon']
    if force_shape == 0 or force_shape == 1:
        shape = shapes[force_shape]
    else:
        shape = shapes[random.randint(0, len(shapes) -1 )]

    sizes = [int(2 + (40 - 2) * i / 9) for i in range(10)]  # 10 sizes from 2 to 40
    size_drawn = sizes[size]
    #big 40x40 medium 20x20 small 10x10
    #Xx (66)
    # Calculate valid x range based on size
    # print(image.size)
    x1 = random.randint(90+size_drawn, 230 - size_drawn)
    y1 = random.randint(148, 179- size_drawn)
    bbox = (x1, y1, x1 + size_drawn, y1 + size_drawn)

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

def add_noise_range_290v2(image, size=0, noise_type='noisy', rgb=False, force_shape=None):
    """
    Same as v1, tries position and validates.
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)
    shapes = ['circle', 'square']
    if force_shape == 0 or force_shape == 1:
        shape = shapes[force_shape]
    else:
        shape = shapes[random.randint(0, len(shapes) - 1)]

    sizes = [int(2 + (40 - 2) * i / 9) for i in range(10)]
    size_drawn = sizes[size]

    brain_mask = np.array(image) > 0
    non_black_coords = np.argwhere(brain_mask)
    min_y, min_x = non_black_coords.min(axis=0)
    max_y, max_x = non_black_coords.max(axis=0)

    for _ in range(50):  # Try up to 50 times to find a valid position
        x1 = random.randint(min_x, max_x - size_drawn)
        y1 = random.randint(min_y, max_y - size_drawn)
        x2, y2 = x1 + size_drawn, y1 + size_drawn

        if (brain_mask[y1:y2, x1:x2].all() and
                x2 <= max_x and y2 <= max_y):  # Ensure the entire shape fits within the brain
            bbox = (x1, y1, x2, y2)
            break
    else:
        raise ValueError("Could not find a valid position within the brain area.")

    if noise_type == 'plain':
        fill_color = 255
        if rgb: fill_color = (255, 255, 255)
    elif noise_type == 'noisy':
        if rgb: fill_color = tuple(random.randint(200, 215) for _ in range(3))
        else: fill_color = random.randint(140, 215)

    if shape == 'circle':
        draw.ellipse(bbox, fill=fill_color)
    elif shape == 'square':
        draw.rectangle(bbox, fill=fill_color)

    return image


def siamese_noise_dataset_fold_range(test_size=0.2, n_splits=5, noise_size=0, resize=True, t1=True, t2=True, random_state=4):
    """
    Creates n folds from the data without duplicating samples and alternates between T1 and T2 image types
    AND range of noise size, with modified add_noise.
    Noise_size : 0 (min) to 9 (max)
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
                    slice1.append(add_noise_range(t1_slice1, size=noise_size))
                    slice2.append(t1_slice2)
                else:
                    slice1.append(t1_slice1)
                    slice2.append(add_noise_range(t1_slice2, size=noise_size))
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
                    slice1.append(add_noise_range(t2_slice1, size=noise_size))
                    slice2.append(t2_slice2)
                else:
                    slice1.append(t2_slice1)
                    slice2.append(add_noise_range(t2_slice2, size=noise_size))
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