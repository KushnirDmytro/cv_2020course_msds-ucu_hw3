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
    thickness = 1
    im = cv2.rectangle(im, (top_left[1], top_left[0]), (bot_right[1], bot_right[0]), color, thickness)
    return im

def cv2_imshow(image, imname='image'):
    cv2.imshow(imname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def NpHisto(srcImage, nBins = 256):
    if len(srcImage.shape) > 2:
        histos = [np.histogram(srcImage[:, :, ch], nBins)[0]
                          for ch in range(srcImage.shape[2])]
        return np.hstack(histos)
    else:
        return np.histogram(srcImage, nBins)[0]


# Constants
EPS = np.finfo(np.float32).eps
HIST_BINS = 16
ALLOWED_ITERATIONS = 4
DROP_EACH_KTH_PIXEL = 2
ALPHA = 0
IMAGE_HISTO_CORRECTION_PERIOD = 5
DISPLAY_FRAME_PERIOD = 100


def ComputeAllImageHistogram(image):
    image_histo = NpHisto(image[:, :, 0], HIST_BINS)  # Ignore last channel for HSV
    histo_sum = np.sum(image_histo)
    image_histo_normalized = (image_histo / histo_sum) + EPS # need this epsillon for further use as denominator.
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
    # with open("data/Panda/Panda/groundtruth_rect.txt") as expected_rects:
    # with open("data/Walking/Walking/groundtruth_rect.txt") as expected_rects:
    # with open("data/Dancer2/Dancer2/groundtruth_rect.txt") as expected_rects:
    # with open("data/Sylvester/Sylvester/groundtruth_rect.txt") as expected_rects:

    # with open("data/Bird1/Bird1/groundtruth_rect.txt") as expected_rects:
    with open("data/RedTeam/RedTeam/groundtruth_rect.txt") as expected_rects:

    # Subj: tracking runner at the competiontions (amongst other runners).
    # Pros: Consistantly colored track at the background.
    #       Scale of the runner itself does not change much.
    #       Position of the runner on a frame does not change much.
    # Cons: Frame itself is moving heavily.
    #       Different bunners come to the frame what changes the scene's histogram heavily.
    # with open("data/Bolt/Bolt/groundtruth_rect.txt") as expected_rects:
        line = expected_rects.readline()
        # TODO: add try-catch for different delimiters
        bbox_value = []
        try:
            bbox_value = [int(val) for val in line.split('\t') if len(val) != 0]
        except ValueError:
            bbox_value = [int(val) for val in line.split(',') if len(val) != 0]
        bb_top_left[1] = bbox_value[0]
        bb_top_left[0] = bbox_value[1]
        bb_width = bbox_value[2]
        bb_height = bbox_value[3]


    dataset = []
    # im_dir_name = "data/Biker/Biker/img"
    # im_dir_name = "data/Panda/Panda/img"
    # im_dir_name = "data/Walking/Walking/img"
    # im_dir_name = "data/Bolt/Bolt/img"
    # im_dir_name = "data/Bird1/Bird1/img"
    im_dir_name = "data/RedTeam/RedTeam/img"
    # im_dir_name = "data/Sylvester/Sylvester/img"
    # im_dir_name = "data/Dancer2/Dancer2/img"
    for impath in tqdm(os.listdir(im_dir_name)):
        image = cv2.imread(join(im_dir_name, impath), cv2.IMREAD_COLOR)
        dataset.append(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        # dataset[-1] = cv2.GaussianBlur(dataset[-1], (13, 13), 0)
    print("Dataset size:", len(dataset))

    im = dataset[0]
    # init stage
    # start_point = (bb_top_left[0], bb_top_left[1])
    # end_point = (start_point[0] + bb_width, start_point[1] + bb_height)

    # Checking initial stage.
    drawn_im = DrawRectOnImage(im,
                               top_left=bb_top_left,
                               bot_right=bb_top_left + np.array([bb_height, bb_width]))
    fig = plt.figure("Initial tracking frame position")
    plt.imshow(drawn_im)
    plt.show()

    image_histo_normalized = ComputeAllImageHistogram(im)
    reference_patch = im[bb_top_left[0]:bb_top_left[0]+bb_height, bb_top_left[1]:bb_top_left[1]+bb_width]
    reference_histo_normalized = ComputeAllImageHistogram(reference_patch)
    reference_likelihood = reference_histo_normalized / image_histo_normalized
    reference_likelihood /= np.sum(reference_likelihood)

    # We use this radius to check potential updated positions for the tracked object
    R = np.sqrt(bb_height * bb_height + bb_width * bb_width) / 2

    for frame_index, new_im in enumerate(dataset[1:]):

        likelihood_im_shape = (new_im.shape[0] - bb_height, new_im.shape[1] - bb_width)
        likelihood_im = np.zeros(likelihood_im_shape)

        bb_top_left_iteration = bb_top_left
        offset = np.array([2, 2]) # initial value for the offset

        # Required stage if image changes over time, per-frame update is not required, could be sparser.
        if frame_index % IMAGE_HISTO_CORRECTION_PERIOD == 0:
            image_histo_normalized = ComputeAllImageHistogram(new_im)

        # Max number of iterations need to be handled, as happens frames with large image artifacts that
        # destroy tracking if allowed to propagate.
        iterations = 0

        # Search for new converged Mean
        while np.linalg.norm(offset) > 0 and iterations < ALLOWED_ITERATIONS:

            r_range = range(int(max(0, bb_top_left_iteration[0] - R)),
                            int(min(likelihood_im_shape[0], bb_top_left_iteration[0] + R)))
            c_range = range(int(max(0, bb_top_left_iteration[1] - R)),
                            int(min(likelihood_im_shape[1], bb_top_left_iteration[1] + R)))

            # for row in tqdm(range(likelihood_im_shape[0])):
            #     for col in range(likelihood_im_shape[1]):

            pix_index = 0
            for row in tqdm(r_range):
                for col in c_range:

                    # Sampling not all histograms inside ROI, mod with prime will give decent semi-random pattern.
                    # For some datasets works worse ("Walking", a.e.)
                    pix_index += 1
                    # if pix_index % DROP_EACH_K_PIXEL == 0:
                    if pix_index % 5 != 0:
                        continue

                    already_computed = likelihood_im[row, col] != 0

                    # Using tracking with circular lookup region to handle cases with narrow frames.
                    dist_to_point = np.linalg.norm(bb_top_left_iteration - np.array([row, col]))
                    if dist_to_point > R or already_computed != 0:
                        continue

                    patch = new_im[row:row+bb_height, col:col+bb_width]
                    patch_histo_normalized = ComputeAllImageHistogram(patch)
                    patch_likelihood = patch_histo_normalized / image_histo_normalized
                    patch_likelihood /= np.sum(patch_likelihood)
                    histo_diff = reference_likelihood - patch_likelihood
                    likelihood = 1 - np.linalg.norm(histo_diff)
                    likelihood_im[row, col] = likelihood

            coords = np.mgrid[r_range.start:r_range.stop, c_range.start: c_range.stop]
            coords = coords.reshape(coords.shape[0], coords.shape[1]*coords.shape[2]).T

            likelihood_reg = likelihood_im[r_range.start:r_range.stop, c_range.start: c_range.stop]
            likelihood_reg = np.reshape(likelihood_reg, (likelihood_reg.shape[0]*likelihood_reg.shape[1]))

            new_top_left = np.average(coords,
                                      weights=likelihood_reg,
                                      axis=0)
            new_top_left = np.round(new_top_left).astype(int)

            # Selection of max value in giver region is also working alternative.
            # If likelihood function is broken will not work either, so usefull for experiments.
            # new_top_left = np.unravel_index(np.argmax(likelihood_im), likelihood_im.shape)

            # Shows computed "costmap" enlarged and with center marked.
            # likelyhood_im_with_point = cv2.circle(likelihood_im, (new_top_left[1], new_top_left[0]), 1, 0)
            # likelihood_reg = likelyhood_im_with_point[r_range.start:r_range.stop, c_range.start: c_range.stop]
            # plt.imshow(likelyhood_reg_with_mean_point)
            # plt.show()

            offset = new_top_left - bb_top_left_iteration
            bb_top_left_iteration = new_top_left
            # print("Offset on iter:", offset)
            iterations += 1

        bb_top_left = bb_top_left_iteration  # update overall hypothesis on border box
        new_reference_patch = new_im[bb_top_left[0]:bb_top_left[0] + bb_height, bb_top_left[1]:bb_top_left[1] + bb_width]
        new_reference_patch_normalized = ComputeAllImageHistogram(new_reference_patch)
        new_reference_patch_normalized /= image_histo_normalized

        # Tracked histogram update stage.
        # Better to avoid absolute change, as jerks in move or recording could cause to total lost of track.
        # For some datasets tracking of reference histo helps, for the others - drift kill the tracking (RedTeam, Walking)
        if ALPHA > 0:
            reference_likelihood = reference_likelihood * (1 - ALPHA) + ALPHA * new_reference_patch_normalized
            reference_likelihood /= np.sum(reference_likelihood)

        if frame_index % DISPLAY_FRAME_PERIOD == 0:
            # Display new position
            drawn_im = DrawRectOnImage(new_im,
                                       top_left=bb_top_left,
                                       bot_right=bb_top_left + np.array([bb_height, bb_width]),
                                       color=(0, 255, 0))
            plt.close()
            fig = plt.figure("Frame {}".format(frame_index))
            plt.imshow(drawn_im[:, :, 2])
            plt.pause(0.1)
            # plt.imshow(likelihood_im)
            # plt.pause(2)

    # Final frame display.
    drawn_im = DrawRectOnImage(new_im,
                               top_left=bb_top_left,
                               bot_right=bb_top_left + np.array([bb_height, bb_width]),
                               color=(0, 255, 1))
    plt.close()
    fig = plt.figure("Frame {}".format(frame_index))
    plt.imshow(drawn_im[:, :, 2])
    plt.show()

if __name__ == "__main__":
    main()