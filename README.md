# A Quantization Loss Compensation Network for Remote Sensing Image Compression

Unofficial Pytorch implementation for the PCS2024 paper 'A Quantization Loss Compensation Network for Remote Sensing Image Compression'.

Note that we don't use Gaussian mixture model mentioned in original paper.

More details can be found in the following paper:
```
@inproceedings{DBLP:conf/pcs/Xiang0W24,
  author       = {Shao Xiang and
                  Jing Xiao and
                  Mi Wang},
  title        = {A Quantization Loss Compensation Network for Remote Sensing Image
                  Compression},
  booktitle    = {Picture Coding Symposium, {PCS} 2024, Taichung, Taiwan, June 12-14,
                  2024},
  pages        = {1--5},
  publisher    = {{IEEE}},
  year         = {2024},
  doi          = {10.1109/PCS60826.2024.10566339},
}
```

# Enviroment
* Python 3.12
* torch 2.4
* Compressai 1.2.6

# Settings

| lambda | N | M                                                                                          |
| ----|------|---------------------------------------------------------------------------------------------|
| 0.0018 | 128 | 192                                                                                    |
| 0.0035 | 128 | 192                                                                                    |
| 0.0067 | 128 | 192                                                                                    |
| 0.0130 | 192 | 320                                                                                    |
| 0.0250 | 192 | 320                                                                                    |
| 0.0483 | 192 | 320                                                                                    |


# Train

```python
accelerate launch train.py \
    --train_dataset 'your/train/dataset' \
    --eval_dataset 'your/test/dataset' \
    --model 'qlc' \
    --log_dir 'your/log/dir' \
    --save_path 'your/checkpoint/dir' \
```