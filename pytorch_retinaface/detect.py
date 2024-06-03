#from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from pytorch_retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from pytorch_retinaface.layers.functions.prior_box import PriorBox
import cv2
from pytorch_retinaface.models.retinaface import RetinaFace
from pytorch_retinaface.utils.box_utils import decode, decode_landm


class Args:
    def __init__(self):
        self.trained_model = "/usr/users/vhassle/psych_track/Pytorch_Retinaface/Resnet50_Final.pth"
        self.cpu = False
        self.confidence_threshold = 0.02
        self.top_k = 5000
        self.nms_threshold = 0.4
        self.keep_top_k = 750
        self.vis_thres = 0.6
        self.network = "resnet50"
args = Args()

CFG_RES50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def load_Retinanet(model_path):
    torch.set_grad_enabled(False)
    cfg = CFG_RES50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, model_path, args.cpu)
    net.eval()
    net = net.to("cuda")
    return net

def process_image(model, image_path, verbose=False):
    """
    Image path can either be a path or a numpy array
    
    """
    cudnn.benchmark = True
    device = "cuda"
    resize = 1

    # testing begin
    if isinstance(image_path , str): 
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    elif isinstance(image_path, np.ndarray):
        img_raw = image_path[:, :, ::-1] # RGB to BGR
    else:
        raise ValueError("Image path must be a string or a numpy array")
    img = np.float32(img_raw)
    if verbose:
        import matplotlib.pyplot as plt
        plt.imshow(img_raw)
        plt.show()

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = model(img)  # forward pass
    priorbox = PriorBox(CFG_RES50, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, CFG_RES50['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, CFG_RES50['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    # show image
    faces = dict()
    for i, b in enumerate(dets):
        if b[4] < args.vis_thres:
            continue
        b = list(map(int, b))
        face_key = f"face_{i}"
        faces[face_key] = (b[0:4])
    return faces




    # for b in dets:
    #     if b[4] < args.vis_thres:
    #         continue
    #     text = "{:.4f}".format(b[4])
    #     b = list(map(int, b))
    #     cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 2)
    #     cx = b[0]
    #     cy = b[1] + 12
    #     cv2.putText(img_raw, text, (cx, cy),
    #                 cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
        # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
        # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
        # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
    # name = "test.jpg"
    # cv2.imwrite(name, img_raw)


# model = load_Retinanet("/usr/users/vhassle/psych_track/Pytorch_Retinaface/Resnet50_Final.pth")
# image_path = "/usr/users/vhassle/datasets/example_images/children/Screenshot 2024-05-30 at 19.50.35.png"
# faces = process_image(image_path, model)
# print(faces)
