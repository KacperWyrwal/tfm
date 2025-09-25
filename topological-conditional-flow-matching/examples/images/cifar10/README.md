# Topological Flow Matching (image generation)

This repository can be used to reproduce the image generation experiments on the CIFAR-10 dataset in Topological Flow Matching.

To reproduce the experiments, start by training the models. 
For the I-CFM and OT-CFM models, please run

```bash
python3 train_cifar10.py --model your_model --seed your_seed
```

for ```your_model``` in ```["icfm", "otcfm"]``` and ```seed``` in ```[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]```.

For the I-TFM and OT-TFM models, please run
```bash
python3 train_cifar10_topological.py --model your_model --seed your_seed --c 0.01
```

for ```your_model``` in ```["cfm_top", "otcfm_top"]``` and ```seed``` in ```[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]```.


Then, evaluate the models. 
For the I-CFM and OT-CFM models, please run

```bash
python3 compute_fid.py --model model_name --seed your_seed 
```

for ```your_model``` in ```["icfm", "otcfm"]``` and ```seed``` in ```[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]```.

For the I-TFM and OT-TFM models, pelase run

```bash
python3 compute_fid_topological.py --model your_model --seed your_seed --c 0.01
```

for ```your_model``` in ```["cfm_top", "otcfm_top"]``` and ```seed``` in ```[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]```.

Thank you!
