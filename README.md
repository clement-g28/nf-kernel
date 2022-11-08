# nf-kernel

## Model codes

The implementation of the different models are based on existing codes:
- RealNVP: https://github.com/ikostrikov/pytorch-flows
- FFJORD: https://github.com/rtqichen/ffjord
- Glow: https://github.com/rosinality/glow-pytorch


## Training

### Toy datasets
Single Moon:
```
python train.py --dataset single_moon --model seqflow --batch_size 100 --lr 0.01 --use_tb --validation 0.1 --set_eigval_manually [50,0.002] --with_noise .1 --fix_mean
```
```
python train.py --dataset single_moon --model ffjord --n_block 1 --dims 64-64-64 --layer_type concatsquash --batch_size 100 --lr 0.01 --use_tb --validation 0.1 --set_eigval_manually [50,0.002] --with_noise .1 --fix_mean
```
Double Moon:
```
python train.py --dataset double_moon --model seqflow --batch_size 100 --lr 0.01 --use_tb --validation 0.1 --set_eigval_manually [50,0.002] --with_noise .1
```
```
python train.py --dataset double_moon --model ffjord --n_block 1 --dims 64-64-64 --layer_type concatsquash --batch_size 100 --lr 0.01 --use_tb --validation 0.1 --set_eigval_manually [50,0.002] --with_noise .1
```
IRIS:
```
python train.py --dataset iris --model seqflow --batch 50 --lr 0.01 --use_tb --validation 0.1 --uniform_eigval --mean_of_eigval 10 --with_noise .2
```
```
python train.py --dataset iris --model ffjord --n_block 1 --dims 64-64-64 --layer_type concatsquash --batch 50 --lr 0.01 --use_tb --validation 0.1 --uniform_eigval --mean_of_eigval 10 --with_noise .2
```
Breast Cancer:
```
python train.py --dataset bcancer --model seqflow --batch 50 --lr 0.005 --use_tb --validation 0.1 --uniform_eigval --mean_of_eigval 10 --with_noise .2
```
```
python train.py --dataset bcancer --model ffjord --n_block 1 --dims 64-64-64 --layer_type concatsquash --batch 50 --lr 0.005 --use_tb --validation 0.1 --uniform_eigval --mean_of_eigval 10 --with_noise .2
```
### Image datasets
MNIST:
```
python train.py --dataset mnist --model cglow --batch 16 --use_tb --validation 0.01 --uniform_eigval --mean_of_eigval 10 --with_noise .5
```

## Find Models

### Classification

Once models have been trained, in order to choose the best model to classify, the following command has to be used by replacing {dataset},{model_type} and {model_name} by your parameters :
```
python get_best_classification.py --folder ./checkpoint/{dataset}/{model_type}/{model_name}
```
For example:
```
python get_best_classification.py --folder ./checkpoint/bcancer/seqflow/f32_nfkernel_lmean1_eigvaluniform10_noise01_dimperlab15
```

### Projection
As pointed out in the article, the evaluation of the model by projection distance is not ideal for image data. However, one can find the best projection model for simpler data using the following command:
```
python get_best_projection.py --folder ./checkpoint/{dataset}/{model_type}/{model_name}
```
For example:
```
python get_best_projection.py --folder ./checkpoint/double_moon/ffjord/b1_nfkernel_lmean1_manualeigval50-0.002_noise01_dimperlab1
```

## Evaluate Models

### Classification
```
python evaluate.py --folder ./checkpoint/{dataset}/{model_type}/{model_name} --eval_type classification --model_to_use classification
```
### Projection
```
python evaluate.py --folder ./checkpoint/{dataset}/{model_type}/{model_name} --eval_type projection --model_to_use projection
```
The evaluation on MNIST for projection can be done on the best classification model by setting model_to_use parameter to classification. 

### MNIST Generation
```
python evaluate.py --folder ./checkpoint/mnist/cglow/{model_name} --eval_type generation --model_to_use classification
```
