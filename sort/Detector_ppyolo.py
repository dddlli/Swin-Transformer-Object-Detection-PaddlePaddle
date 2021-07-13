import time
import yaml
from functools import reduce
from preprocess_img import *
import os
import numpy as np
import paddle.fluid as fluid


class Detector(object):
    """
    Args:
        config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of __model__, __params__ and infer_cfg.yml
        use_gpu (bool): whether use gpu
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16)
        threshold (float): threshold to reserve the result for output.
    """

    def __init__(self,
                 config,
                 model_dir,
                 use_gpu=False,
                 run_mode='fluid',
                 threshold=0.5):
        self.config = config
        if self.config.use_python_inference:
            self.executor, self.program, self.fecth_targets = load_executor(
                model_dir, use_gpu=use_gpu)
        else:
            self.predictor = load_predictor(
                model_dir,
                run_mode=run_mode,
                min_subgraph_size=self.config.min_subgraph_size,
                use_gpu=use_gpu)

    def preprocess(self, im):
        preprocess_ops = []
        for op_info in self.config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            if op_type == 'Resize':
                new_op_info['arch'] = self.config.arch
            preprocess_ops.append(eval(op_type)(**new_op_info))
        im, im_info = preprocess(im, preprocess_ops)
        inputs = create_inputs(im, im_info, self.config.arch)
        return inputs, im_info

    def postprocess(self, np_boxes, np_masks, np_lmk, im_info, threshold=0.5):
        # postprocess output of predictor
        results = {}
        detections = []
        if np_lmk is not None:
            results['landmark'] = lmk2out(np_boxes, np_lmk, im_info, threshold)

        expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
        np_boxes = np_boxes[expect_boxes, :]
        """for box in np_boxes:
            print('class_id:{:d}, confidence:{:.4f},'
                  'left_top:[{:.2f},{:.2f}],'
                  ' right_bottom:[{:.2f},{:.2f}]'.format(
                      int(box[0]), box[1], box[2], box[3], box[4], box[5]))"""
        results['boxes'] = np_boxes
        if np_masks is not None:
            np_masks = np_masks[expect_boxes, :, :, :]
            results['masks'] = np_masks
            
        for box in np_boxes:
            detection = (box[0], box[1], box[2], box[3], box[4], box[5])
            detections.append(detection)

        return results, detections

    def predict(self,
                image,
                threshold=0.5,
                warmup=0,
                repeats=1,
                run_benchmark=False):
        '''
        Args:
            image (str/np.ndarray): path of image/ np.ndarray read by cv2
            threshold (float): threshold of predicted box' score
        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's results include 'masks': np.ndarray:
                            shape:[N, class_num, mask_resolution, mask_resolution]
        '''
        inputs, im_info = self.preprocess(image)
        np_boxes, np_masks, np_lmk = None, None, None
        if self.config.use_python_inference:
            for i in range(warmup):
                outs = self.executor.run(self.program,
                                         feed=inputs,
                                         fetch_list=self.fecth_targets,
                                         return_numpy=False)
            t1 = time.time()
            for i in range(repeats):
                outs = self.executor.run(self.program,
                                         feed=inputs,
                                         fetch_list=self.fecth_targets,
                                         return_numpy=False)
            t2 = time.time()
            ms = (t2 - t1) * 1000.0 / repeats
            print("Inference: {} ms per image".format(ms))
            np_boxes = np.array(outs[0])
            if self.config.mask_resolution is not None:
                np_masks = np.array(outs[1])
        else:
            input_names = self.predictor.get_input_names()
            for i in range(len(input_names)):
                input_tensor = self.predictor.get_input_tensor(input_names[i])
                input_tensor.copy_from_cpu(inputs[input_names[i]])

            for i in range(warmup):
                self.predictor.zero_copy_run()
                output_names = self.predictor.get_output_names()
                boxes_tensor = self.predictor.get_output_tensor(output_names[0])
                np_boxes = boxes_tensor.copy_to_cpu()
                if self.config.mask_resolution is not None:
                    masks_tensor = self.predictor.get_output_tensor(output_names[1])
                    np_masks = masks_tensor.copy_to_cpu()

                if self.config.with_lmk is not None and self.config.with_lmk == True:
                    face_index = self.predictor.get_output_tensor(output_names[
                        1])
                    landmark = self.predictor.get_output_tensor(output_names[2])
                    prior_boxes = self.predictor.get_output_tensor(output_names[
                        3])
                    np_face_index = face_index.copy_to_cpu()
                    np_prior_boxes = prior_boxes.copy_to_cpu()
                    np_landmark = landmark.copy_to_cpu()
                    np_lmk = [np_face_index, np_landmark, np_prior_boxes]

            t1 = time.time()
            for i in range(repeats):
                self.predictor.zero_copy_run()
                output_names = self.predictor.get_output_names()
                boxes_tensor = self.predictor.get_output_tensor(output_names[0])
                np_boxes = boxes_tensor.copy_to_cpu()
                if self.config.mask_resolution is not None:
                    masks_tensor = self.predictor.get_output_tensor(output_names[1])
                    np_masks = masks_tensor.copy_to_cpu()

                if self.config.with_lmk is not None and self.config.with_lmk == True:
                    face_index = self.predictor.get_output_tensor(output_names[
                        1])
                    landmark = self.predictor.get_output_tensor(output_names[2])
                    prior_boxes = self.predictor.get_output_tensor(output_names[
                        3])
                    np_face_index = face_index.copy_to_cpu()
                    np_prior_boxes = prior_boxes.copy_to_cpu()
                    np_landmark = landmark.copy_to_cpu()
                    np_lmk = [np_face_index, np_landmark, np_prior_boxes]
            t2 = time.time()
            ms = (t2 - t1) * 1000.0 / repeats
            print("Inference: {} ms per batch image".format(ms))

        # do not perform postprocess in benchmark mode
        results = []
        if not run_benchmark:
            if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
                print('[WARNNING] No object detected.')
                results = {'boxes': np.array([])}
            else:
                results, detections = self.postprocess(np_boxes, np_masks, np_lmk, im_info, threshold=threshold)

        return results, detections


def create_inputs(im, im_info, model_arch='YOLO'):
    """generate input for different model type
    Args:
        im (np.ndarray): image (np.ndarray)
        im_info (dict): info of image
        model_arch (str): model type
    Returns:
        inputs (dict): input of model
    """
    inputs = {}
    inputs['image'] = im
    origin_shape = list(im_info['origin_shape'])
    resize_shape = list(im_info['resize_shape'])
    pad_shape = list(im_info['pad_shape']) if im_info[
        'pad_shape'] is not None else list(im_info['resize_shape'])
    scale_x, scale_y = im_info['scale']
    if 'YOLO' in model_arch:
        im_size = np.array([origin_shape]).astype('int32')
        inputs['im_size'] = im_size
    
    return inputs


class Config():

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.use_python_inference = yml_conf['use_python_inference']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask_resolution = None
        if 'mask_resolution' in yml_conf:
            self.mask_resolution = yml_conf['mask_resolution']
        self.with_lmk = None
        if 'with_lmk' in yml_conf:
            self.with_lmk = yml_conf['with_lmk']


def load_predictor(model_dir,
                   run_mode='fluid',
                   batch_size=1,
                   use_gpu=False,
                   min_subgraph_size=3):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        use_gpu (bool): whether use gpu
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need use_gpu == True.
    """
    if not use_gpu and not run_mode == 'fluid':
        raise ValueError("Predict by TensorRT mode: {}, expect use_gpu==True, but use_gpu == {}".format(run_mode, use_gpu))
    if run_mode == 'trt_int8':
        raise ValueError("TensorRT int8 mode is not supported now, "
                         "please use trt_fp32 or trt_fp16 instead.")
    precision_map = {
        'trt_int8': fluid.core.AnalysisConfig.Precision.Int8,
        'trt_fp32': fluid.core.AnalysisConfig.Precision.Float32,
        'trt_fp16': fluid.core.AnalysisConfig.Precision.Half
    }
    config = fluid.core.AnalysisConfig(
        os.path.join(model_dir, '__model__'),
        os.path.join(model_dir, '__params__'))
    if use_gpu:
        # initial GPU memory(M), device ID
        config.enable_use_gpu(100, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    else:
        config.disable_gpu()

    if run_mode in precision_map.keys():
        config.enable_tensorrt_engine(
            workspace_size=1 << 10,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[run_mode],
            use_static=False,
            use_calib_mode=False)

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    predictor = fluid.core.create_paddle_predictor(config)
    return predictor


def load_executor(model_dir, use_gpu=False):
    if use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    program, feed_names, fetch_targets = fluid.io.load_inference_model(
        dirname=model_dir,
        executor=exe,
        model_filename='__model__',
        params_filename='__params__')
    return exe, program, fetch_targets
