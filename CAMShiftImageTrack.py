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

# Constants
EPS = np.finfo(np.float32).eps

# Hyperparameters : knobs for tuning collected in 1 place.
HIST_BINS = 16
ALLOWED_ITERATIONS = 4
DROP_EACH_KTH_PIXEL = 2
ALPHA = 0.0
IMAGE_HISTO_CORRECTION_PERIOD = 5
DISPLAY_FRAME_PERIOD = 1
N_CHANNELS_USED = 1


def draw_rect_on_image(im, top_left, bot_right, color=(0, 255, 0)):
    thickness = 1
    im = cv2.rectangle(im, (top_left[1], top_left[0]), (bot_right[1], bot_right[0]), color, thickness)
    return im


def cv2_imshow(image, imname='image'):
    cv2.imshow(imname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def np_histo(srcImage, nBins=256):
    if len(srcImage.shape) > 2:
        histos = [np.histogram(srcImage[:, :, ch], nBins)[0]
                  for ch in range(srcImage.shape[2])]
        return np.hstack(histos)
    else:
        return np.histogram(srcImage, nBins)[0]


def read_init_data():
    im_dir_name = ""
    expected_border_box_path = ""

    bb_top_left = [-1, -1]
    bb_width = -1
    bb_height = -1
    bb_ratio = -1
    bb_requires_init = True

    # Expected image path and path to file where bounding box is defined
    if len(sys.argv) == 3:  # 1 path for input images and 1 path for expected border boxes
        print("Reading image path and path to border boxes")
        im_dir_name = sys.argv[1]
        expected_border_box_path = sys.argv[2]

    elif len(sys.argv) == 6:  # case of input data path and 4 values for border box
        print("Reading image path and border box values")
        im_dir_name = sys.argv[1]
        bb_top_left[1] = int(sys.argv[2])
        bb_top_left[0] = int(sys.argv[3])
        bb_width = int(sys.argv[4])
        bb_height = int(sys.argv[5])
        bb_requires_init = False

    else:
        print("Input format is unknown, supported formats are:"
              "\n[data path] and [filename with border boxes] "
              "\n[data path] [border_box_top_left_col] [border_box_top_left_row] [border_box_width] [border_box_height]"
              "\n\nDefault preconfigured demo will be launched \n\n")
        # im_dir_name = "data/Biker/Biker/img"
        # im_dir_name = "data/Panda/Panda/img"
        # im_dir_name = "data/Walking/Walking/img"
        # im_dir_name = "data/Bolt/Bolt/img"
        # im_dir_name = "data/Bird1/Bird1/img"
        im_dir_name = "data/RedTeam/RedTeam/img"
        # im_dir_name = "data/Sylvester/Sylvester/img"
        # im_dir_name = "data/Dancer2/Dancer2/img"

        # Subj: tracking the cyclists helmet.
        # Brief: pros: overall scene is static, scene histo is not changed much.
        # Cons: Patch background changes from sky to grass, this transition is dangerous for algo.
        #       Once cyclist rotates helmet color change jumps dramatically.
        #       Cyclist's head is increasing in size.
        #       The helmet itself is colored like a sky, so it's importance is dumped.
        #       There is a killing moment when tracked head goes out of the frame at all
        # expected_border_box_path = "data/Biker/Biker/groundtruth_rect.txt"

        # Subj: tracking runner at the competitions (amongst other runners).
        # Pros: Consistently colored track at the background.
        #       Scale of the runner itself does not change much.
        #       Position of the runner on a frame does not change much.
        # Cons: Frame itself is moving heavily.
        #       Different banners come to the frame what changes the scene's histogram heavily.
        # expected_border_box_path = "data/Bolt/Bolt/groundtruth_rect.txt"
        # expected_border_box_path = "data/Panda/Panda/groundtruth_rect.txt"
        # expected_border_box_path = "data/Walking/Walking/groundtruth_rect.txt"
        # expected_border_box_path = "data/Dancer2/Dancer2/groundtruth_rect.txt"
        # expected_border_box_path = "data/Sylvester/Sylvester/groundtruth_rect.txt"
        # expected_border_box_path = "data/Bird1/Bird1/groundtruth_rect.txt"
        expected_border_box_path = "data/RedTeam/RedTeam/groundtruth_rect.txt"

    if bb_requires_init:
        with open(expected_border_box_path) as expected_rects:
            line = expected_rects.readline()
            try:
                bbox_value = [int(val) for val in line.split('\t') if len(val) != 0]
            except ValueError:
                bbox_value = [int(val) for val in line.split(',') if len(val) != 0]
            bb_top_left[1] = bbox_value[0]
            bb_top_left[0] = bbox_value[1]
            bb_width = bbox_value[2]
            bb_height = bbox_value[3]

    print(" Using border box with top_left at [{}], width=[{}] height=[{}] \n".format(bb_top_left, bb_width, bb_height))
    return im_dir_name, bb_top_left, bb_width, bb_height


def compute_image_histogram(image):
    image_histo = np_histo(image[:, :, :N_CHANNELS_USED], HIST_BINS)  # Ignore last channel for HSV
    histo_sum = np.sum(image_histo)
    image_histo_normalized = (image_histo / histo_sum) + EPS  # need this epsilon for further use as denominator.
    return image_histo_normalized


def main():
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))

    im_dir_name, bb_top_left, bb_width, bb_height = read_init_data()
    compensation_for_dropped_pixels = 1 / (1 - 1 / DROP_EACH_KTH_PIXEL)

    dataset = []
    print("Using images from ", im_dir_name)
    for impath in tqdm(os.listdir(im_dir_name)):
        image = cv2.imread(join(im_dir_name, impath), cv2.IMREAD_COLOR)
        dataset.append(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        # dataset[-1] = cv2.GaussianBlur(dataset[-1], (13, 13), 0)
    print("Dataset size:", len(dataset))

    im = dataset[0]
    # init stage

    # Checking initial stage.
    drawn_im = draw_rect_on_image(im,
                                  top_left=bb_top_left,
                                  bot_right=bb_top_left + np.array([bb_height, bb_width]))
    fig = plt.figure("Initial tracking frame position")
    plt.imshow(drawn_im)
    plt.show()

    image_histo_normalized = compute_image_histogram(im)
    reference_patch = im[bb_top_left[0]:bb_top_left[0] + bb_height, bb_top_left[1]:bb_top_left[1] + bb_width]
    reference_histo_normalized = compute_image_histogram(reference_patch)
    reference_likelihood = reference_histo_normalized / image_histo_normalized
    reference_likelihood /= np.sum(reference_likelihood)

    # We use this radius to check potential updated positions for the tracked object
    R = np.sqrt(bb_height * bb_height + bb_width * bb_width) / 2

    stored_bbox_sizes = []

    for frame_index, new_im in tqdm(enumerate(dataset[1:])):

        likelihood_im_shape = (new_im.shape[0] - bb_height, new_im.shape[1] - bb_width)
        likelihoods_im = np.zeros(likelihood_im_shape)

        bb_top_left_iteration = bb_top_left
        offset = np.array([2, 2])  # Some initial value for the offset

        # Required stage if image changes over time, per-frame update is not required, could be sparser.
        if frame_index % IMAGE_HISTO_CORRECTION_PERIOD == 0:
            image_histo_normalized = compute_image_histogram(new_im)

        # Max number of iterations need to be handled, as happens frames with large image artifacts that
        # destroy tracking if allowed to propagate.
        iterations = 0

        # Search for new converged Mean
        while np.linalg.norm(offset) > 0 and iterations < ALLOWED_ITERATIONS:

            r_range = range(int(max(0, bb_top_left_iteration[0] - R)),
                            int(min(likelihood_im_shape[0], bb_top_left_iteration[0] + R)))
            c_range = range(int(max(0, bb_top_left_iteration[1] - R)),
                            int(min(likelihood_im_shape[1], bb_top_left_iteration[1] + R)))

            pix_index = 0
            # for row in tqdm(r_range):
            for row in r_range:
                for col in c_range:

                    # Sampling not all histograms inside ROI, mod with prime will give decent semi-random pattern.
                    # For some datasets works worse ("Walking", a.e.)
                    pix_index += 1
                    if pix_index % DROP_EACH_KTH_PIXEL == 0:
                        continue

                    already_computed = likelihoods_im[row, col] != 0

                    # Using tracking with circular lookup region to handle cases with narrow frames.
                    dist_to_point = np.linalg.norm(bb_top_left_iteration - np.array([row, col]))
                    if dist_to_point > R or already_computed != 0:
                        continue

                    patch = new_im[row:row + bb_height, col:col + bb_width]
                    patch_histo_normalized = compute_image_histogram(patch)
                    patch_likelihood = patch_histo_normalized / image_histo_normalized
                    patch_likelihood /= np.sum(patch_likelihood)
                    histo_diff = reference_likelihood - patch_likelihood
                    likelihood = 1 - np.linalg.norm(histo_diff)
                    likelihoods_im[row, col] = likelihood

            coordinates = np.mgrid[r_range.start:r_range.stop, c_range.start: c_range.stop]
            coordinates = coordinates.reshape(coordinates.shape[0], coordinates.shape[1] * coordinates.shape[2]).T

            likelihood_reg = likelihoods_im[r_range.start:r_range.stop, c_range.start: c_range.stop]
            likelihood_reg = np.reshape(likelihood_reg, (likelihood_reg.shape[0] * likelihood_reg.shape[1]))

            new_top_left = np.average(coordinates,
                                      weights=likelihood_reg,
                                      axis=0)
            new_top_left = np.round(new_top_left).astype(int)

            # Selection of max value in giver region is also working alternative.
            # If likelihood function is broken will not work either, so useful for experiments.
            # new_top_left = np.unravel_index(np.argmax(likelihoods_im), likelihoods_im.shape)

            # Shows computed "costmap" enlarged and with center marked.
            # likelihood_im_with_point = cv2.circle(likelihoods_im, (new_top_left[1], new_top_left[0]), 1, 0)
            # likelihood_reg = likelihood_im_with_point[r_range.start:r_range.stop, c_range.start: c_range.stop]
            # plt.imshow(likelihood_reg_with_mean_point)
            # plt.show()

            offset = new_top_left - bb_top_left_iteration
            bb_top_left_iteration = new_top_left
            # print("Offset on iter:", offset)
            iterations += 1

        # CAMShift stage of adding positioning to the barycenter:
        # if frame_index % 10 == 0:

        bb_ratio = bb_height / bb_width
        r, c = bb_top_left_iteration

        bb_top_left = bb_top_left_iteration  # update overall hypothesis on border box

        likelihood_region = likelihoods_im[r - bb_height//2: r + bb_height//2,
                                           c - bb_width//2: c + bb_width//2]
        M00 = np.sum(likelihood_region) #* compensation_for_dropped_pixels
        bb_width_recommended = np.round(2 * np.sqrt(M00)).astype(int)
        bb_height_recommended = np.round(bb_width_recommended * bb_ratio).astype(int)
        stored_bbox_sizes.append([bb_width_recommended, bb_height_recommended])
        if len(stored_bbox_sizes) > 5:
            recommended_track_window_sizes = np.mean(stored_bbox_sizes, axis=0)
            bb_width2, bb_height2 = np.round(recommended_track_window_sizes).astype(int)
            bb_top_left_offset = [(bb_width2 - bb_width) // 2, (bb_height2 - bb_height) // 2]
            bb_top_left -= bb_top_left_offset
            bb_top_left = [max(bb_top_left[0], 0), max(bb_top_left[1], 0)]
            bb_width, bb_height = bb_width2, bb_height2
            stored_bbox_sizes = stored_bbox_sizes[2:]

        # r, c = bb_top_left_iteration
        # r -= bb_height//2
        # c -= bb_width//2
        # likelihood_region = likelihoods_im[r:r + bb_height, c:c + bb_width]
        # coordinates = np.mgrid[r:r + bb_height, c:c + bb_width]
        # M00 = np.sum(likelihood_region)
        # M10 = np.sum(coordinates[0, :, :] * likelihood_region)
        # M01 = np.sum(coordinates[1, :, :] * likelihood_region)
        # baricenter_row = M10 / M00
        # baricenter_col = M01 / M00
        # bb_top_left = np.round([baricenter_row, baricenter_col]).astype(int)
        # print(max([bb_top_left_iteration - bb_top_left))
        # print("baricenter correction : ", bb_top_left - bb_top_left_iteration)
        # likelihood_im_with_points = cv2.circle(likelihood_region,
        #                                        (new_top_left[1] - bb_top_left_iteration[1], new_top_left[0] - bb_top_left_iteration[0]),
        #                                        1,
        #                                        0)
        # likelihood_im_with_points = cv2.circle(likelihood_region,
        #                                        (new_top_left[1] - bb_top_left_iteration[1],
        #                                         new_top_left[0] - bb_top_left_iteration[0]),
        #                                        1,
        #                                        0)
        # plt.imshow(likelihood_region)
        # plt.show()
        # M01 = np.sum(patch * coordinates)


        new_reference_patch = new_im[bb_top_left[0]:bb_top_left[0] + bb_height,
                                     bb_top_left[1]:bb_top_left[1] + bb_width]
        new_reference_patch_normalized = compute_image_histogram(new_reference_patch)
        new_reference_patch_normalized /= image_histo_normalized

        # Tracked histogram update stage.
        # Better to avoid absolute change, as jerks in move or recording could cause to total lost of track.
        # For some datasets tracking of reference histo helps, for others - drift kill the tracking (RedTeam, Walking)
        if ALPHA > 0:
            reference_likelihood = reference_likelihood * (1 - ALPHA) + ALPHA * new_reference_patch_normalized
            reference_likelihood /= np.sum(reference_likelihood)

        if frame_index % DISPLAY_FRAME_PERIOD == 0:
            # Display new position
            drawn_im = draw_rect_on_image(new_im,
                                          top_left=bb_top_left,
                                          bot_right=bb_top_left + np.array([bb_height, bb_width]),
                                          color=(0, 255, 0))
            # plt.close()
            # fig = plt.figure("Frame {}".format(frame_index))
            plt.imshow(drawn_im[:, :, 1])
            plt.pause(0.2)
            # plt.imshow(likelihoods_im)
            # plt.pause(2)

    # Final frame display.
    drawn_im = draw_rect_on_image(new_im,
                                  top_left=bb_top_left,
                                  bot_right=bb_top_left + np.array([bb_height, bb_width]),
                                  color=(0, 255, 1))
    plt.close()
    fig = plt.figure("Frame {}".format(frame_index))
    plt.imshow(drawn_im[:, :, 2])
    plt.show()


if __name__ == "__main__":
    main()
