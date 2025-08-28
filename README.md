# NIMBLE4SimHand

This project utilizes a parameterized hand model, **NIMBLE**, for generating continuous and dense simulated right-hand datasets including meshes, textures, and keypoints, to support downstream tasks such as **hand pose estimation**. The NIMBLE model is based on a non-rigid hand structure with bones and muscles, enabling realistic hand motion simulation and anatomically plausible deformation.

## Usage

1. `demo4sequence.py`: Generates a sequence of hand meshes with continuously changing poses, starting from the rest pose principal components (all zeros) and randomly varying the 30-dimensional PCA-reduced pose parameters (originally 20x3=60 dimensions)

    ```python
    python demo4sequence.py
    ```

2. `demo_ik.py`: Takes 25 hand keypoints in the NIMBLE format as input and performs inverse kinematics (IK) to compute the local axis-angle transformation parameters from the rest joints to the target joints. This can be used to drive the NIMBLE hand model, enabling a pipeline of keypoints (e.g., captured from motion capture systems) → IK solving → NIMBLE hand model driving → mesh generation for arbitrary hand gestures.

    ```python
    python demo_ik.py
    ```

The IK solver for the NIMBLE model is implemented in `utils.py`. The file `utils_for_mano.py` provides IK solvers for both MANO and NIMBLE models. These files can be directly integrated into other projects (note: the order of axis-angle vectors differs from the MANO definition).

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
