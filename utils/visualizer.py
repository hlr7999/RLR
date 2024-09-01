import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def denormalization(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = np.array(mean)
    std = np.array(std)
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


class Visualizer(object):
    def __init__(self, root, prefix=''):
        self.root = root
        self.prefix = prefix
        os.makedirs(self.root, exist_ok=True)
    
    def set_prefix(self, prefix):
        self.prefix = prefix

    def plot(self, test_imgs, scores, gt_masks, file_names, img_anomalies, decode_list):
        """
        Args:
            test_imgs (ndarray): shape (N, 3, h, w)
            scores (ndarray): shape (N, h, w)
            img_scores (ndarray): shape (N, )
            gt_masks (ndarray): shape (N, 1, h, w)
        """
        vmax = scores.max() * 255.
        vmin = scores.min() * 255. + 10
        vmax = vmax - 220
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for i in range(len(scores)):
            img = test_imgs[i]
            img = denormalization(img)
            gt_mask = gt_masks[i].squeeze()
            score = scores[i]
            #score = gaussian_filter(score, sigma=4)
            heat_map = score * 255
            if len(decode_list) > 0:
                fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 3))
            else:
                fig_img, ax_img = plt.subplots(1, 3, figsize=(9, 3))

            fig_img.subplots_adjust(wspace=0.05, hspace=0)
            for ax_i in ax_img:
                ax_i.axes.xaxis.set_visible(False)
                ax_i.axes.yaxis.set_visible(False)

            ax_img[0].imshow(img)
            ax_img[0].title.set_text('Input image')
            ax_img[1].imshow(gt_mask, cmap='gray')
            ax_img[1].title.set_text('GroundTruth')
            ax_img[2].imshow(heat_map, cmap='jet', norm=norm, interpolation='none')
            ax_img[2].imshow(img, cmap='gray', alpha=0.7, interpolation='none')
            ax_img[2].title.set_text('Segmentation')
            if len(decode_list) > 0:
                decode_img = decode_list[i]
                decode_img = denormalization(decode_img)
                ax_img[3].imshow(decode_img)
                ax_img[3].title.set_text('Decode image')
            
            save_path = os.path.join(self.root, img_anomalies[i])
            os.makedirs(save_path, exist_ok=True)
            fig_img.savefig(os.path.join(save_path, file_names[i]), dpi=300)

            plt.close()
