from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import numpy as np
import time
import torch
import math

from models.model import create_model, load_model
from utils.image import get_affine_transform, affine_transform
from utils.image import draw_umich_gaussian, gaussian_radius
from utils.debugger import Debugger
from utils.tracker import Tracker
from datasets.dataset_factory import get_dataset


class BaseDetector(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')
    self.model = create_model(opt.arch, opt.heads, opt.head_conv)
    self.model = load_model(self.model, opt.load_model)
    self.model = self.model.to(opt.device)
    self.model.eval()

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = True
    self.rest_focal_length = opt.rest_focal_length
    self.flip_idx = opt.flip_idx
    self.cnt = 0
    self.pre_images = None
    self.pre_image_ori = None
    self.tracker = Tracker(opt)
    self.debugger = Debugger(opt=opt, dataset=opt.dataset)

  def pre_process(self, image, scale, input_meta={}):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      inp_height = (new_height | self.opt.pad) + 1
      inp_width = (new_width | self.opt.pad) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    out_height =  inp_height // self.opt.down_ratio
    out_width =  inp_width // self.opt.down_ratio
    trans_output = get_affine_transform(c, s, 0, [out_width, out_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'calib': np.array(input_meta['calib'], dtype=np.float32) \
             if 'calib' in input_meta else \
             self._get_default_calib(width, height)}
    meta.update({'c': c, 's': s, 'height': height, 'width': width,
            'out_height': out_height, 'out_width': out_width,
            'inp_height': inp_height, 'inp_width': inp_width,
            'trans_input': trans_input, 'trans_output': trans_output})
    if 'pre_dets' in input_meta:
      meta['pre_dets'] = input_meta['pre_dets']
    if 'cur_dets' in input_meta:
      meta['cur_dets'] = input_meta['cur_dets']
    return images, meta

  def process(self, images, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image, results):
   raise NotImplementedError

  def run(self, image_or_path_or_tensor, meta={}):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, track_time, display_time, tot_time = 0, 0, 0, 0
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                        theme=self.opt.debugger_theme, opt=self.opt)
    start_time = time.time()
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''): 
      image = cv2.imread(image_or_path_or_tensor)
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True

    loaded_time = time.time()
    load_time += (loaded_time - start_time)
    
    detections = []
    for scale in self.scales:
      scale_start_time = time.time()
      if not pre_processed:
        images, meta = self.pre_process(image, scale, meta)
      else:
        # import pdb; pdb.set_trace()
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
        if 'pre_dets' in pre_processed_images['meta']:
          meta['pre_dets'] = pre_processed_images['meta']['pre_dets']
        if 'cur_dets' in pre_processed_images['meta']:
          meta['cur_dets'] = pre_processed_images['meta']['cur_dets']

      images = images.to(self.opt.device)

      # initializing tracker
      pre_hms, pre_inds = None, None
      if self.opt.tracking:
        # initialize the first frame
        if self.pre_images is None:
          print('Initialize tracking!')
          self.pre_images = images
          self.tracker.init_track(
            meta['pre_dets'] if 'pre_dets' in meta else [], image)
        if self.opt.pre_hm:
          # render input heatmap from tracker status
          # pre_inds is not used in the current version.
          # We used pre_inds for learning an offset from previous image to
          # the current image.
          pre_hms, pre_inds = self._get_additional_inputs(
            self.tracker.tracks, meta, with_hm=not self.opt.zero_pre_hm)

      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time

      output, dets, forward_time = self.process(images, self.pre_images, pre_hms, pre_inds, return_time=True)

      net_time += forward_time - pre_process_time
      decode_time = time.time()
      dec_time += decode_time - forward_time
      
      if self.opt.debug >= 2:
        self.debug(debugger, images, dets, output, scale)
      
      dets = self.post_process(dets, meta, scale)
      post_process_time = time.time()
      post_time += post_process_time - decode_time

      detections.append(dets)
    
    results = self.merge_outputs(detections)
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time

    if self.opt.tracking and 'tracking' in output:
      # public detection mode in MOT challenge
      public_det = meta['cur_dets'] if self.opt.public_det else None
      # add tracking id to results
      results = self.tracker.step(results, public_det)
      self.pre_images = images

    tracking_time = time.time()
    track_time += tracking_time - end_time
    tot_time += tracking_time - start_time

    if self.opt.debug >= 1:
      self.show_results(self.debugger, image, results)
      
    show_results_time = time.time()
    display_time += show_results_time - end_time

    ret = {'results': results, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time, 'track': track_time,
            'display': display_time}

    if self.opt.save_video:
      try:
        # return debug image for saving video
        ret.update({'generic': self.debugger.imgs['generic']})
      except Exception as e:
        print(e)

    return ret
  
  def _get_additional_inputs(self, dets, meta, with_hm=True):
    '''
    Render input heatmap from previous trackings.
    '''
    trans_input, trans_output = meta['trans_input'], meta['trans_output']
    inp_width, inp_height = meta['inp_width'], meta['inp_height']
    out_width, out_height = meta['out_width'], meta['out_height']
    input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32)

    output_inds = []
    for det in dets:
      if det['score'] < self.opt.pre_thresh or det['active'] == 0:
        continue
      bbox = self._trans_bbox(det['bbox'], trans_input, inp_width, inp_height)
      bbox_out = self._trans_bbox(
        det['bbox'], trans_output, out_width, out_height)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if (h > 0 and w > 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 10], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        if with_hm:
          draw_umich_gaussian(input_hm[0], ct_int, radius)
        ct_out = np.array(
          [(bbox_out[0] + bbox_out[2]) / 2, 
           (bbox_out[1] + bbox_out[3]) / 10], dtype=np.int32)
        output_inds.append(ct_out[1] * out_width + ct_out[0])
    if with_hm:
      input_hm = input_hm[np.newaxis]
      if self.opt.flip_test:
        input_hm = np.concatenate((input_hm, input_hm[:, :, :, ::-1]), axis=0)
      input_hm = torch.from_numpy(input_hm).to(self.opt.device)
    output_inds = np.array(output_inds, np.int64).reshape(1, -1)
    output_inds = torch.from_numpy(output_inds).to(self.opt.device)
    return input_hm, output_inds
  
  def _trans_bbox(self, bbox, trans, width, height):
    '''
    Transform bounding boxes according to image crop.
    '''
    bbox = np.array(copy.deepcopy(bbox), dtype=np.float32)
    bbox[:2] = affine_transform(bbox[:2], trans)
    bbox[2:] = affine_transform(bbox[2:], trans)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
    return bbox

  def _get_default_calib(self, width, height):
    calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib


  def _sigmoid_output(self, output):
    if 'hm' in output:
      output['hm'] = output['hm'].sigmoid_()
    if 'hm_hp' in output:
      output['hm_hp'] = output['hm_hp'].sigmoid_()
    if 'dep' in output:
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
      output['dep'] *= self.opt.depth_scale
    return output


  def _flip_output(self, output):
    average_flips = ['hm', 'wh', 'dep', 'dim']
    neg_average_flips = ['amodel_offset']
    single_flips = ['ltrb', 'nuscenes_att', 'velocity', 'ltrb_amodal', 'reg',
      'hp_offset', 'rot', 'tracking', 'pre_hm']
    for head in output:
      if head in average_flips:
        output[head] = (output[head][0:1] + flip_tensor(output[head][1:2])) / 2
      if head in neg_average_flips:
        flipped_tensor = flip_tensor(output[head][1:2])
        flipped_tensor[:, 0::2] *= -1
        output[head] = (output[head][0:1] + flipped_tensor) / 2
      if head in single_flips:
        output[head] = output[head][0:1]
      if head == 'hps':
        output['hps'] = (output['hps'][0:1] + 
          flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
      if head == 'hm_hp':
        output['hm_hp'] = (output['hm_hp'][0:1] + \
          flip_lr(output['hm_hp'][1:2], self.flip_idx)) / 2

    return output


  def reset_tracking(self):
    self.tracker.reset()
    self.pre_images = None
    self.pre_image_ori = None
