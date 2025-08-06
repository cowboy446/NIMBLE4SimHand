# NIMBLE4SimHand

This project utilizes a parameterized hand model, **NIMBLE**, for generating continuous and dense simulated right-hand datasets including meshes, textures, and keypoints, to support downstream tasks such as **hand pose estimation**. The NIMBLE model is based on a non-rigid hand structure with bones and muscles, enabling realistic hand motion simulation and anatomically plausible deformation.

## Usage
```python
python demo.py
```


## 📚 Based on NIMBLE

This work builds upon the following paper:

> **NIMBLE: A Non-Rigid Hand Model with Bones and Muscles**  
> Yuwei Li, Longwen Zhang, Zesong Qiu, Yingwenqi Jiang, Nianyi Li, Yuexin Ma, Yuyao Zhang, Lan Xu, and Jingyi Yu.  
> *ACM Transactions on Graphics (SIGGRAPH 2022)*  
> [DOI: 10.1145/3528223.3530079](https://doi.org/10.1145/3528223.3530079)

Their original repository can be found [here](https://github.com/reyuwei/NIMBLE_model).

If you use this model in your research, please consider citing the original paper:

```bibtex
@article{li2022nimble,
  author = {Li, Yuwei and Zhang, Longwen and Qiu, Zesong and Jiang, Yingwenqi and Li, Nianyi and Ma, Yuexin and Zhang, Yuyao and Xu, Lan and Yu, Jingyi},
  title = {NIMBLE: A Non-Rigid Hand Model with Bones and Muscles},
  year = {2022},
  issue_date = {July 2022},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {41},
  number = {4},
  issn = {0730-0301},
  url = {https://doi.org/10.1145/3528223.3530079},
  doi = {10.1145/3528223.3530079},
  journal = {ACM Trans. Graph.},
  month = {jul},
  articleno = {120},
  numpages = {16}
}
