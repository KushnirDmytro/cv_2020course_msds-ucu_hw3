import os
import numpy as np
import pandas as pd
import cv2
import sys
import bisect
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from tqdm import trange, tqdm
from sklearn.cluster import KMeans


# uses simple radial kernel
def MeanShiftIteration(lkh, values, counts, domain_span, kernel_r, init_pos, verbose=True):

    # for row in range(

    kernel_lower_lim = np.maximum(domain_span[0][0], init_pos[0] - kernel_r)
    kernel_upper_lim = np.minimum(domain_span[1][0], init_pos[0] + kernel_r)

    bounding_box_low_index = bisect.bisect_left(values[:, 0], kernel_lower_lim)
    bounding_box_hight_index = bisect.bisect_right(values[:, 0], kernel_upper_lim)

    if verbose:
        print("KernelLowerLimit = ", kernel_lower_lim)
        print("KernelUpeprLimit = ", kernel_upper_lim)
        print("clusterCenter =", init_pos)
        print("lowerLimFirstDim =", bounding_box_low_index)
        print("upperLimFirstDim =", bounding_box_hight_index)
        print("usedCluster R = ", kernel_r)

    n_points = 0
    sum_points = np.zeros_like(values[0])

    n_points2 = 0
    sum_points2 = np.zeros_like(values[0])

    kernel_r_squared = kernel_r * kernel_r

    selector = [(p_val - init_pos) @ (p_val - init_pos) < kernel_r_squared
                for p_val
                in values[bounding_box_low_index:bounding_box_hight_index]]

    selected_p = counts[bounding_box_low_index:bounding_box_hight_index][selector]
    selected_v = values[bounding_box_low_index:bounding_box_hight_index][selector]
    n_points = np.sum(selected_p)

    sum_points = selected_p @ selected_v

    # Note: used type in64 is large enough as accumulator for given dataset.
    if n_points != 0:
        new_mean_point = sum_points / n_points
    else:
        new_mean_point = init_pos

    if verbose:
        print("pointsInSpan =", n_points)
        print("accum = ", sum_points)
        print("meanPoints =", new_mean_point)

    return new_mean_point

def DrawRectOnImage(im, top_left, bot_right, color=(0, 255, 0)):
    thickness = 2
    im = cv2.rectangle(im, top_left, bot_right, color, thickness)
    return im

def cv2_imshow(image, imname='image'):
    cv2.imshow(imname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def NpHisto(srcImage, nBins = 256):
    if len(srcImage.shape) > 2:
        return np.hstack([np.histogram(srcImage[:, :, ch], nBins, range=(0, nBins-1))[0]
            for ch in range(srcImage.shape[2])])
    else:
        return np.histogram(srcImage, nBins, range=(0, nBins-1))[0]


# Constants
EPS = np.finfo(np.float32).eps
HIST_BINS = 100

def ComputeAllImageHistogram(image):
    image_histo = NpHisto(image, HIST_BINS)  # Ignore last channel for HSV
    image_histo_normalized = (image_histo / np.sum(image_histo)) + EPS
    return image_histo_normalized

def main():
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))

    # TODO: Read border box for tracking region
    # TODO: Read path for dataset

    bb_top_left = [0, 0]
    bb_width = -1
    bb_height = -1
    print(os.getcwd())

    # Subj: tracking the cyclists helmet.
    # Brief: pros: overall scene is static, scene histo is not changed much.
    # Cons: Patch background changes from sky to grass, this transition is dangerous for algo.
    #       Once cyclist rotates helmet color change jumps dramatically.
    #       Cyclist's head is increasing in size.
    #       The helmet itself is colored like a sky, so it's importance is dumped.
    #       There is a killing moment when tracked head goes out of the frame at all.
    # with open("data/Biker/Biker/groundtruth_rect.txt") as expected_rects:

    # Subj: tracking runner at the competiontions (amongst other runners).
    # Pros: Consistantly colored track at the background.
    #       Scale of the runner itself does not change much.
    #       Position of the runner on a frame does not change much.
    # Cons: Frame itself is moving heavily.
    #       Different bunners come to the frame what changes the scene's histogram heavily.
    with open("data/Bolt/Bolt/groundtruth_rect.txt") as expected_rects:
        line = expected_rects.readline()
        bboxValue = [ int(val) for val in line.split(',') if len(val) != 0]
        bb_top_left[0] = bboxValue[0]
        bb_top_left[1] = bboxValue[1]
        bb_width = bboxValue[2]
        bb_height = bboxValue[3]


    dataset = []
    # im_dir_name = "data/Biker/Biker/img"
    im_dir_name = "data/Bolt/Bolt/img"
    for impath in tqdm(os.listdir(im_dir_name)):
        image = cv2.imread(join(im_dir_name, impath), cv2.IMREAD_COLOR)
        dataset.append(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    print("Dataset size:", len(dataset))

    im = dataset[0]
    # init stage
    start_point = (bb_top_left[0], bb_top_left[1])
    end_point = (start_point[0] + bb_width, start_point[1] + bb_height)

    # Checking initial stage.
    drawn_im = DrawRectOnImage(im,
                               top_left=(bb_top_left[0], bb_top_left[1]),
                               bot_right=(bb_top_left[0] + bb_width, bb_top_left[1] + bb_height))
    plt.imshow(drawn_im)
    plt.show()

    image_histo_normalized = ComputeAllImageHistogram(im)

    reference_patch = im[bb_top_left[1]:bb_top_left[1]+bb_height, bb_top_left[0]:bb_top_left[0]+bb_width]
    reference_histo_normalized = ComputeAllImageHistogram(reference_patch)
    reference_histo_normalized /= image_histo_normalized
    reference_histo_normalized /= np.sum(reference_histo_normalized) + EPS

    old_center = np.round((np.array(start_point) + np.array(end_point)) / 2)

    # We use this radius to check potential updated positions for the tracked object
    R = np.sqrt(bb_height * bb_height + bb_width + bb_width)
    for new_im in dataset[1:]:
        likelyhood_im_shape = (im.shape[0] - bb_height, im.shape[1] - bb_width)
        likelyhood_im = np.zeros(likelyhood_im_shape)
        c_range = range(int(max(0, old_center[0] - R / 2)), int(min(likelyhood_im_shape[1], old_center[0] + R / 2)))
        r_range = range(int(max(0, old_center[1] - R / 2)), int(min(likelyhood_im_shape[0], old_center[1] + R / 2)))

        # for row in tqdm(range(likelyhood_im_shape[0])):
        for row in tqdm(r_range):
            # for col in range(likelyhood_im_shape[1]):
            for col in c_range:
                patch = new_im[row:row+bb_height, col:col+bb_width]
                patch_histo_normalized = ComputeAllImageHistogram(patch)
                patch_histo_normalized /= image_histo_normalized
                patch_histo_normalized /= (np.sum(patch_histo_normalized) + EPS)
                histo_diff = reference_histo_normalized - patch_histo_normalized
                likelyhood = 1 - np.linalg.norm(histo_diff)
                likelyhood_im[row, col] = likelyhood

        # TODO: compute weights without allocation of full-size image.
        weights = likelyhood_im[r_range.start:r_range.stop - 1, c_range.start: c_range.stop - 1]

        coords = np.mgrid[r_range.start:r_range.stop - 1, c_range.start: c_range.stop - 1]
        coords = coords.reshape(coords.shape[0], coords.shape[1]*coords.shape[2]).T
        weights = np.reshape(weights, (weights.shape[0]*weights.shape[1]))
        new_mean = np.average(coords,
                              weights=weights,
                              axis=0)
        new_mean = np.round(new_mean).astype(int)

        old_center = old_center.astype(int)

        offset = new_mean - [old_center[1], old_center[0]]
        old_center = np.array([new_mean[1], new_mean[0]])

        patch = new_im[row:row + bb_height, col:col + bb_width]
        patch_histo_normalized = ComputeAllImageHistogram(patch)
        image_histo_normalized = ComputeAllImageHistogram(new_im)

        # Tracked histogram update stage.
        # Better to avoid absolute change, as jerks in move could cause to total lost of track.
        reference_histo_normalized = patch_histo_normalized / image_histo_normalized
        reference_histo_normalized /= np.sum(reference_histo_normalized) + EPS

        # Update old im
        # im = new_im

        drawn_im = DrawRectOnImage(new_im,
                                   top_left=(bb_top_left[0], bb_top_left[1]),
                                   bot_right=(bb_top_left[0] + bb_width, bb_top_left[1] + bb_height),
                                   color=(0, 0, 255))
        bb_top_left += offset
        drawn_im = DrawRectOnImage(drawn_im,
                                   top_left=(bb_top_left[0], bb_top_left[1]),
                                   bot_right=(bb_top_left[0] + bb_width, bb_top_left[1] + bb_height),
                                   color=(0, 255, 0))
        plt.imshow(drawn_im)
        plt.pause(0.001)

if __name__ == "__main__":
    main()