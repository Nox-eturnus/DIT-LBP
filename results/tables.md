## Within-condition accuracy (preliminary trial run, 1 seed)

| Method | realistic | misaligned | relief_0.03 |
|---|---|---|---|
| LBP (1 image) | 0.733 | 0.790 | 0.963 |
| LBP (4 images) | 0.923 | **0.989** | 0.955 |
| LDB (1 image) | 0.622 | 0.673 | 0.602 |
| LDB (4 images) | 0.926 | 0.844 | 0.628 |
| Image difference (4 images) | **1.000** | 0.957 | **0.977** |
| DIT-LBP (proposed) | **1.000** | 0.903 | 0.807 |
| DIT-LBP multi-radius | **1.000** | 0.955 | 0.923 |
| DIT-LBP + LBP (1 image) | **1.000** | 0.938 | 0.903 |

## Macro F1, within-condition (mean over seeds)

| Method | realistic | misaligned | relief_0.03 |
|---|---|---|---|
| LBP (1 image) | 0.728 | 0.789 | 0.963 |
| LBP (4 images) | 0.923 | **0.989** | 0.955 |
| LDB (1 image) | 0.621 | 0.673 | 0.600 |
| LDB (4 images) | 0.926 | 0.842 | 0.627 |
| Image difference (4 images) | **1.000** | 0.957 | **0.977** |
| DIT-LBP (proposed) | **1.000** | 0.903 | 0.805 |
| DIT-LBP multi-radius | **1.000** | 0.954 | 0.923 |
| DIT-LBP + LBP (1 image) | **1.000** | 0.937 | 0.902 |

## Trained on realistic, tested on stress conditions (accuracy)

| Method | misaligned | relief_0.03 |
|---|---|---|
| LBP (1 image) | 0.793 | **0.810** |
| LBP (4 images) | **0.923** | 0.719 |
| LDB (1 image) | 0.639 | 0.560 |
| LDB (4 images) | 0.889 | 0.531 |
| Image difference (4 images) | 0.773 | 0.500 |
| DIT-LBP (proposed) | 0.608 | 0.500 |
| DIT-LBP multi-radius | 0.523 | 0.514 |
| DIT-LBP + LBP (1 image) | 0.722 | 0.500 |

## Feature extraction time per 128x128 patch stack (ms, median)

| Method | ms |
|---|---|
| LBP (1 image) | 0.24 |
| LBP (4 images) | 0.96 |
| LDB (1 image) | 1.71 |
| LDB (4 images) | 6.67 |
| Image difference (4 images) | 0.65 |
| DIT-LBP (proposed) | 0.89 |
| DIT-LBP multi-radius | 2.67 |
| DIT-LBP + LBP (1 image) | 1.12 |

## Mean LBP bit flips per pixel (of 8)

| condition   |   label |   flip_h |   flip_v |
|:------------|--------:|---------:|---------:|
| misaligned  |       0 |    5.029 |    4.328 |
| misaligned  |       1 |    3.275 |    3.28  |
| realistic   |       0 |    6.355 |    5.928 |
| realistic   |       1 |    0.644 |    0.663 |
| relief_0.03 |       0 |    1.921 |    1.815 |
| relief_0.03 |       1 |    1.453 |    1.434 |

## Real flat control: PhoTex white card ('reg')

|   slant | method                      |   frac_called_flat |   flip_h |   flip_v |
|--------:|:----------------------------|-------------------:|---------:|---------:|
|      45 | LBP (1 image)               |              0     |    3.856 |    3.865 |
|      45 | LBP (4 images)              |              0     |    3.856 |    3.865 |
|      45 | LDB (1 image)               |              0.625 |    3.856 |    3.865 |
|      45 | LDB (4 images)              |              0.938 |    3.856 |    3.865 |
|      45 | Image difference (4 images) |              1     |    3.856 |    3.865 |
|      45 | DIT-LBP (proposed)          |              0     |    3.856 |    3.865 |
|      45 | DIT-LBP multi-radius        |              0     |    3.856 |    3.865 |
|      45 | DIT-LBP + LBP (1 image)     |              0     |    3.856 |    3.865 |
