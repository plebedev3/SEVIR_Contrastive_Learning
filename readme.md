# Contrastive Learning with SEVIR

## Data preprocessing

### Normed Gray Images

Using the ipython file ConvertToNormedCmaps.ipynb, will convert a sevir dtype to batched pickle files with the required OpenCV gray scale for optical flow calculation

### Optical Flows

Using LLMapReduce, the map_to_opt_flow and the corrsibonding submit sh script will map along nodes and calculate the optical flow. The reduce step will later combine the batched pickle files together, but this is not nessisary as of now

Using the command:
LLMapReduce --mapper map_to_opt_flow.sh --reducer reduce_to_opt_flow.sh --input SEVIR_IR069_NORMED_CMAPS --output SEVIR_IR069_OPT_FLOWS --slotsPerTask=16 --np=7 --prefix=ir069 --ext=noext --ndata=4

## NCE Loss function implmentation
The submodule Pytorch-NCE is needed https://github.com/Stonesjtu/Pytorch-NCE
