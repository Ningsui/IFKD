This is the code for the ACM MM2024 paper "Information Fusion with Knowledge Distillation for Fine-grained Remote Sensing Object Detection"

Our codes are based on MMDetection. Please follow the installation of MMDetection and make sure you can run it successfully.

Add and Replace the codes Add the configs/. in our codes to the configs/ in mmdetectin's codes. Add the mmdet/models/detectors/. in our codes to the mmdet/models/detectors/.

## Citation

If you find our repo useful for your research, please cite us:

```
@inproceedings{10.1145/3664647.3681684,
author = {Zhang, Sheng and Yang, Xi},
title = {Information Fusion with Knowledge Distillation for Fine-grained Remote Sensing Object Detection},
year = {2024},
isbn = {9798400706868},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3664647.3681684},
doi = {10.1145/3664647.3681684},
pages = {8081–8089},
numpages = {9},
keywords = {fine-grained object detection, information fusion., knowledge distillation, remote sensing images},
location = {Melbourne VIC, Australia},
series = {MM '24}
}
```

This project is based on the open source codebase [MMDetection](https://github.com/open-mmlab/mmdetection).
```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
