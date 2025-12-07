### ForestSplats

[WACV'26] ForestSplats: Deformable transient field for Gaussian Splatting in the Wild [[ArXiv](https://arxiv.org/abs/2503.06179)]   [[Project Page](https://kalelpark.github.io/ForestSplats/)]
<div align="center">
<img width="525" height="267" alt="ForestSplat_Poster_Size" src="assets/updated_poster.png" />
</div>

### Cloning the Repository
```bash
git clone https://github.com/kalelpark/ForestSplats.git --recursive
```

### Installation
Follow the instructions in the [WildGaussians](https://github.com/jkulhanek/wild-gaussians/) to setup the environment. To download datasets and split train/test, please follow the [NeRFBaselines](https://nerfbaselines.github.io/).

### Train
```bash
python scripts.py
```
To understand the our superpixel-based masking strategy more easily, please refer to [mask_module.py](./mask_module.py). For the uncertainty-aware density control, please also refer to [submodules](./submodules).
> Note that the implementation includes only the masking and densification strategies. \
> If you have any questions or issues during the reimplementation, please feel free to reach out.

### Apperance Variation
For Appearance Variation, modify the line 535 of [evaluation.py](./ForestSplats/evaluation.py).

### Acknowledgements
We thank the following projects for enabling and motivating ForestSplats:
- [WildGaussians](https://github.com/jkulhanek/wild-gaussians)
- [SuperPixels](https://arxiv.org/abs/2003.12929)
- [PixelGS](https://link.springer.com/chapter/10.1007/978-3-031-72655-2_19)
