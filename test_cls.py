"""
Test script for classification and end-to-end module.
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import time
import torch
import numpy as np


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.imp_weight = False
    opt.batch_size = 1
    opt.num_threads = 0             # test code only supports num_threads = 0
    opt.serial_batches = True       # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True              # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1             # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)   # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)       # create a model given opt.model and other options
    model.setup(opt)                # regular setup: load and print networks; create schedulers

    save_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    if opt.load_iter > 0:  # load_iter is 0 by default
        save_dir = '{:s}_iter{:d}'.format(save_dir, opt.load_iter)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    sum_loadtime = 0
    sum_runtime = 0
    sum_posttime = 0
    sum_alltime = 0
    correct = 0
    tricorrect = 0
    result = []
    for r in range(1):
        num_img = 0
        for i, data in enumerate(dataset):
            start_time = time.time()    # timer for computation per iteration
            model.set_input(data)       # unpack data from data loader
            
            load_time = time.time()

            model.test()                # run inference
            output, label = model.get_current_output()
            
            end_time = time.time()

            img_path = model.get_image_paths()

            sum_loadtime += load_time - start_time
            sum_runtime += end_time - load_time

            gt = label
            pred = torch.argmax(output, dim=1)
                
            num_img += len(img_path)
            pred = pred.detach().cpu().numpy()
            gt = gt.detach().cpu().numpy()
            for p, g, i_path in zip(pred, gt, img_path):
                p = p.item()
                g = g.item()
                if p == g:
                    correct += 1
                if np.abs(p-g) <= 1:
                    tricorrect += 1
            if r == 0:
                result.append([pred.item(), gt.item(), img_path[0]])
            
            post_time = time.time()
            sum_posttime += post_time - end_time
            sum_alltime += post_time - start_time

        rate = correct / num_img
        print('Prediction accuracy: %f' % (rate * 100 / (r+1)))
        trirate = tricorrect / num_img
        print('Prediction Tri accuracy: %f' % (trirate * 100 / (r+1)))
    
    infer_loadtime = sum_loadtime / num_img / (r+1)
    infer_runtime = sum_runtime / num_img / (r+1)
    infer_posttime = sum_posttime / num_img / (r+1)
    print('Loading time, Sum: %.3f, Avg: %.3f' % (sum_loadtime, infer_loadtime))
    print('Processing time, Sum: %.3f, Avg: %.3f' % (sum_runtime, infer_runtime))
    print('Post calculating time, Sum: %.3f, Avg: %.3f' % (sum_posttime, infer_posttime))
    print('All time used: %.3f' % (sum_alltime / num_img / (r+1)))
    print('Number of images per second: %s' % str(np.round(1/(sum_alltime / num_img / (r+1)))))
    
    with open("%s/pred_%s.txt" % (save_dir, opt.name), "w") as f:
        f.write('Prediction accuracy: %f \n' % (rate * 100))
        f.write('Prediction accuracy: %f \n' % (trirate * 100))
        f.write('Loading time, Sum: %.3f, Avg: %.3f \n' % (sum_loadtime, infer_loadtime))
        f.write('Processing time, Sum: %.3f, Avg: %.3f \n' % (sum_runtime, infer_runtime))
        f.write('Post calculating time, Sum: %.3f, Avg: %.3f \n' % (sum_posttime, infer_posttime))
        f.write('All time used: %.3f \n' % (sum_alltime / num_img / (r+1)))
        f.write('Number of images per second: %s \n' % str(np.round(1/(sum_alltime / num_img / (r+1)))))
        f.write('[Predicted Label, GT Label, Image Name] \n')
        for r in result:
            f.write(str(r) +"\n")
