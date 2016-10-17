import os
import sys
import cPickle
import numpy as np

Interval = 10

FEATURE_PATH = 'feature' #feature/HVC999702.mp4/0001.pkl
POOLING_PATH = '%d-mean-pooling'%Interval #10-mean-pooling

os.mkdir(POOLING_PATH)
#for video in os.listdir(FRAME_PATH):
for video in open('MED12.txt', 'r'):
  video = video.strip()

  print video
  sys.stdout.flush()

  video_feature_path = os.path.join(FEATURE_PATH, video)
  video_pooling_file = os.path.join(POOLING_PATH, '%s.pkl'%video.split('.')[0]) #10-mean-pooling/HVC999702.pkl

  data = []
  cnt = 0
  cycle = 0.0
  tmp = 0.0
  while True:
    feature_path = os.path.join(video_feature_path, '%04d.pkl'%(cnt+1)) 
    if os.path.isfile(feature_path):
      cnt += 1
      cycle += 1
      with open(feature_path, 'rb') as f:
        tmp += cPickle.load(f)
      if cycle == Interval:
        data.append(tmp/cycle)
        cycle = 0.0
        tmp = 0.0
    else:
      break
  with open(video_pooling_file, 'wb') as f:
    cPickle.dump(data, f)


