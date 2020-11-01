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
  return np.hstack([np.histogram(srcImage[:, :, 0], nBins, range=(0, nBins-1))[0],
                    np.histogram(srcImage[:, :, 1], nBins, range=(0, nBins-1))[0],
                    np.histogram(srcImage[:, :, 2], nBins, range=(0, nBins-1))[0]])

def main():
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))

    # TODO: Read border box for tracking region
    # TODO: Read path for dataset

    # for j in range(0, 3):
    #     img = np.random.normal(size=(100, 150))
    #     plt.figure(1);
    #     plt.clf()
    #     plt.imshow(img)
    #     plt.title('Number ' + str(j))
    #     plt.pause(3)

    bb_top_left = [0, 0]
    bb_width = -1
    bb_height = -1
    print(os.getcwd())
    # with open("data/Biker/Biker/groundtruth_rect.txt") as expected_rects:
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
        dataset.append(cv2.imread(join(im_dir_name, impath), cv2.IMREAD_COLOR))
    print("Dataset size:", len(dataset))

    # dataset = [cv2.cvtColor(im, cv2.COLOR) for im in dataset]

    im = dataset[0]

    # Constants
    EPS = np.finfo(np.float32).eps
    HIST_BINS = 100

    # init stage
    start_point = (bb_top_left[0], bb_top_left[1])
    end_point = (start_point[0] + bb_width, start_point[1] + bb_height)
    color = (0, 255, 0)
    thickness = 2
    im = cv2.rectangle(im, start_point, end_point, color, thickness)
    drawn_im = DrawRectOnImage(im,
                               top_left=(bb_top_left[0], bb_top_left[1]),
                               bot_right=(bb_top_left[0] + bb_width, bb_top_left[1] + bb_height))

    plt.imshow(drawn_im)
    plt.show()

    image_histo = NpHisto(im, HIST_BINS)
    image_histo_normalized = image_histo / np.sum(image_histo)

    reference_histo = NpHisto(im[bb_top_left[1]:bb_top_left[1]+bb_height, bb_top_left[0]:bb_top_left[0]+bb_width], HIST_BINS)
    reference_histo_normalized = reference_histo / np.sum(reference_histo)

    old_center = np.round((np.array(start_point) + np.array(end_point)) / 2)
    R = np.sqrt(bb_height * bb_height + bb_width + bb_width)
    for new_im in dataset[1:]:
        likelyhood_im_shape = (im.shape[0] - bb_height, im.shape[1] - bb_width)
        likelyhood_im = np.zeros(likelyhood_im_shape)

        c_range = range(int(max(0, old_center[0] - R/2)), int(min(likelyhood_im.shape[0], old_center[0] + R/2)))
        r_range = range(int(max(0, old_center[1] - R / 2)), int(min(likelyhood_im.shape[1], old_center[1] + R / 2)))

        # for row in tqdm(range(likelyhood_im_shape[0])):
        for row in tqdm(r_range):
            # for col in range(likelyhood_im_shape[1]):
            for col in c_range:
                patch = new_im[row:row+bb_height, col:col+bb_width]
                patch_histo = NpHisto(patch, HIST_BINS)
                histo_sum = np.sum(patch_histo) + EPS
                patch_histo_normalized = patch_histo / histo_sum
                histo_diff = reference_histo_normalized - patch_histo_normalized
                likelyhood = 1 - np.linalg.norm(histo_diff)
                likelyhood_im[row, col] = likelyhood

        # plt.imshow(likelyhood_im)
        # plt.show()

        r_range_list = list(r_range)
        c_range_list = list(c_range)
        weights = likelyhood_im[r_range_list[0]:r_range_list[-1], c_range_list[0]:c_range_list[-1]]

        # TODO: replace with some build-in function call
        coords = np.zeros( (likelyhood_im.shape[0], likelyhood_im.shape[1], 3) )
        for r in r_range:
            for c in c_range:
                coords[r, c, 0] = r
                coords[r, c, 1] = c

        coords = coords[r_range_list[0]:r_range_list[-1], c_range_list[0]:c_range_list[-1], :2]
        coords = np.reshape(coords, (coords.shape[0]*coords.shape[1], 2))
        weights = np.reshape(weights, (weights.shape[0]*weights.shape[1]))
        new_mean = np.average(coords,
                              weights=weights,
                              axis=0)
        new_mean = np.round(new_mean).astype(int)

        print("new mean shape:", new_mean.shape)
        print("new mean:", new_mean)

        old_center = old_center.astype(int)

        offset = new_mean - [old_center[1], old_center[0]]
        old_center = np.array([new_mean[1], new_mean[0]])
        im = new_im
        drawn_im = DrawRectOnImage(im,
                                   top_left=(bb_top_left[0], bb_top_left[1]),
                                   bot_right=(bb_top_left[0] + bb_width, bb_top_left[1] + bb_height),
                                   color=(0, 0, 255))

        bb_top_left += offset
        drawn_im = DrawRectOnImage(drawn_im,
                                   top_left=(bb_top_left[0], bb_top_left[1]),
                                   bot_right=(bb_top_left[0] + bb_width, bb_top_left[1] + bb_height),
                                   color=(0, 255, 0))
        plt.imshow(drawn_im)
        plt.pause(0.1)

        # TODO: Perform tracking

if __name__ == "__main__":
    main()