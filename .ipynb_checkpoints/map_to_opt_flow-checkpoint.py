import os
import h5py # needs conda/pip install h5py
import matplotlib.pyplot as plt
import sys
from sevir.display import get_cmap
import cv2
import re
import numpy as np
import matplotlib
import glob
from tqdm import tqdm

inOutFileList = open(sys.argv[1],"rb")
print(inOutFileList.read())
OPTICAL_FLOW_CALC_METHOD = cv2.DualTVL1OpticalFlow_create()
def calc_optical_flow(sevir_np_data):
    flows = []
    for index in range(len(sevir_np_data)-1):
        flow = OPTICAL_FLOW_CALC_METHOD.calc(sevir_np_data[index],sevir_np_data[index+1], None)
        flows.append(flow)
    return np.array(flows)

for line in inOutFileList.readlines():
    (inFile,outFile) = line.split()
    print("Reading from " + inFile + " and writing to " + outFile)
    with open(inFile, "rb") as pickle_file:
        data = pickle.load(pickle_file)
    opt_flows = []
    for curr_event_id, curr_np_data_arr in data:
        opt_flows.append((curr_event_id, calc_optical_flow(curr_np_data_arr)))
    with open(outFile + "_flow.pickle", "wb") as pickle_file:
        pickle.dump(opt_flows, pickle_file)