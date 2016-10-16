import sys
import os
import cv2
import numpy as np
import caffe
import cPickle

MODEL = '/home/hehe/feature-models/VGG19/VGG_ILSVRC_19_layers.caffemodel'
PROTOTXT  = '/home/hehe/feature-models/VGG19/VGG_ILSVRC_19_layers_deploy.prototxt'

FRAME_PATH = '/home/hehe/cvpr/extract/frame'
FEATURE_PATH = 'feature'

# Build up VGG
net = caffe.Net(PROTOTXT, MODEL, caffe.TEST)

caffe.set_mode_gpu()
caffe.set_device(0)

batch_size = 1
net.blobs['data'].reshape(batch_size, 3, 224, 224)

# Set up input
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('/opt/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

# Extract feature
#for video in os.listdir(FRAME_PATH):
for video in open('MED12.txt', 'r'):
  video = video.strip()
  
  print video
  sys.stdout.flush()

  video_frame_path = os.path.join(FRAME_PATH, video)
  video_feature_path = os.path.join(FEATURE_PATH, video)
  os.makedirs(video_feature_path)
  for frame in os.listdir(video_frame_path):
    frame_path = os.path.join(video_frame_path, frame)
    feature_path = os.path.join(video_feature_path, '%s.pkl'%(frame.split('.')[0]))

    try:
      image = caffe.io.load_image(frame_path)
      net.blobs['data'].data[...] = transformer.preprocess('data', image)
    except Exception as e:
      print e
      continue
    else:
      feature = net.forward(blobs=['fc7'])['fc7'][0]
      with open(feature_path, 'wb') as f:
        cPickle.dump(feature, f)
