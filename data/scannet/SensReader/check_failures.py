import os
import numpy as np
import subprocess
import matplotlib.pyplot as plt

CHECK_DIR = '/mnt/Data/Datasets/ScanNet_v2/scans/'
SENS_DATA_DIR = '/mnt/Data/Datasets/ScanNet_v1/'
SCAN_NAMES = [line.rstrip() for line in open('/mnt/Data/Datasets/ScanNet_v1/sceneid_sort.txt')]

num_frames_statistics = []
for scan_id, scan_name in enumerate(SCAN_NAMES):
    img_folder = os.path.join(CHECK_DIR, scan_name, 'color')
    if os.path.exists(img_folder):
        list = os.listdir(img_folder)  # dir is your directory path
        num_frames = len(list)
        print(img_folder, num_frames)
        num_frames_statistics.append(num_frames)
    else:
        print('Warning: Fail to extract frames from ', scan_name)
        os.remove(os.path.join(SENS_DATA_DIR, 'scans', scan_name, '{0}.sens'.format(scan_name)))
        subprocess.call(["python2", "download-scannet.py", "--o", '/mnt/Data/Datasets/ScanNet_v1/',
                         "--id", scan_name, "--type", ".sens"])

num_frames_statistics = np.array(num_frames_statistics)
plt.hist(num_frames_statistics, 20)
plt.show()
print("================================")
print('median number of frames is ', np.median(num_frames_statistics))
print('mean number of frames is ', np.mean(num_frames_statistics))
print('minimum number of frames is ', np.min(num_frames_statistics))
print('maximum number of frames is ', np.max(num_frames_statistics))