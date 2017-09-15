# GIF Super-Resolution

## Download Giphy Data

**`data/dnld_giphy.py`:** Download GIF data from https://giphy.com/. Usage as below. The GIFs tagged 'face' will be downloaded into `data/raw_gifs/face/`.
```
python dnld_giphy.py [tag]
python dnld_giphy.py 'face'
```

## Run Code

**`code/gif2img.py`:** Convert GIFs into PNG images. Convert all GIFs tagged 'face' in `data/raw_gifs/` to PNG images and store into `data/raw_imgs/`.
```
python gif2img.py
```

**`gen_hr_lr.py`:** Generate high-resolution (GT) and low-resolution (test input) images. HR images are stored in `data/hr_imgs/` and LR images are stored in `data/lr_imgs/`
```
python gen_hr_lr.py
```

**`bicu_inter.py`:** Generate Bicubic Interpolation of low-resolution images. BI images are stored in `data/bi_imgs/`
```
python bicu_inter.py
```
