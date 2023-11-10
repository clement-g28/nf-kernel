
# Machine Learning without the pre-image problem thanks to normalizing flow

## Model codes

The implementation of the different models are based on existing codes:
- RealNVP: https://github.com/ikostrikov/pytorch-flows
- FFJORD: https://github.com/rtqichen/ffjord
- Glow: https://github.com/rosinality/glow-pytorch
- MoFlow : https://github.com/calvin-zcx/moflow


## Training

### Classifications & Denoising

#### Toy datasets

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
python train.py --dataset iris --model seqflow --batch_size 50 --lr 0.01 --use_tb --validation 0.1 --uniform_eigval --mean_of_eigval 10 --with_noise .2
```
```
python train.py --dataset iris --model ffjord --n_block 1 --dims 64-64-64 --layer_type concatsquash --batch_size 50 --lr 0.01 --use_tb --validation 0.1 --uniform_eigval --mean_of_eigval 10 --with_noise .2
```
Breast Cancer:
```
python train.py --dataset bcancer --model seqflow --batch_size 50 --lr 0.005 --use_tb --validation 0.1 --uniform_eigval --mean_of_eigval 10 --with_noise .2
```
```
python train.py --dataset bcancer --model ffjord --n_block 1 --dims 64-64-64 --layer_type concatsquash --batch_size 50 --lr 0.005 --use_tb --validation 0.1 --uniform_eigval --mean_of_eigval 10 --with_noise .2
```

#### Image datasets

MNIST:
```
python train.py --dataset mnist --model cglow --batch_size 16 --use_tb --validation 0.01 --uniform_eigval --mean_of_eigval 10 --with_noise .5
```

#### Graph datasets

MUTAG:
```
python train.py --dataset MUTAG --model moflow --n_flow 32 --n_block 1 --batch_size 10 --lr 0.0002 --noise_scale 0.6 --use_tb --validation 0.1 --uniform_eigval --beta 200 --mean_of_eigval 1.5 --n_epoch 3000 --save_each_epoch 10 --split_graph_dim
```

Letter-med:
```
python train.py --dataset Letter-med --model moflow --n_flow 32 --n_block 1 --batch_size 200 --lr 0.001 --noise_scale 0.6 --noise_scale_x 0.2 --use_tb --validation 0.1 --uniform_eigval --beta 200 --mean_of_eigval 20 --n_epoch 10000 --save_each_epoch 10 --split_graph_dim
```

### Regression
Swiss roll:
```
python train.py --dataset swissroll --model seqflow --batch_size 20 --lr 0.01 --use_tb --validation 0.1 --uniform_eigval --isotrope_gaussian --beta 50 --mean_of_eigval 10 --with_noise 0.1 
```
```
python train.py --dataset swissroll --model ffjord --n_block 1 --dims 64-64-64 --layer_type concatsquash --batch_size 20 --lr 0.01 --use_tb --validation 0.1 --uniform_eigval --isotrope_gaussian --beta 50 --mean_of_eigval 10 --with_noise 0.1 
```

Diabetes:
```
python train.py --dataset diabetes --model seqflow --batch_size 20 --lr 0.01 --use_tb --validation 0.1 --uniform_eigval --isotrope_gaussian --beta 50 --mean_of_eigval 0.1
```
```
python train.py --dataset diabetes --model ffjord --n_block 1 --dims 64-64-64 --layer_type concatsquash --batch_size 20 --lr 0.01 --use_tb --validation 0.1 --uniform_eigval --isotrope_gaussian --beta 50 --mean_of_eigval 0.1
```

QSAR aquatic toxicity:
```
python train.py --dataset aquatoxi --model seqflow --batch_size 20 --lr 0.001 --use_tb --validation 0.1 --uniform_eigval --isotrope_gaussian --beta 200 --mean_of_eigval 0.01
```
```
python train.py --dataset aquatoxi --model ffjord --n_block 1 --dims 64-64-64 --layer_type concatsquash --batch_size 20 --lr 0.001 --use_tb --validation 0.1 --uniform_eigval --isotrope_gaussian --beta 50 --mean_of_eigval 0.1
```

QSAR fish toxicity:
```
python train.py --dataset fishtoxi --model seqflow --batch_size 20 --lr 0.001 --use_tb --validation 0.1 --uniform_eigval --isotrope_gaussian --beta 50 --mean_of_eigval 0.1
```
```
python train.py --dataset fishtoxi --model ffjord --n_block 1 --dims 64-64-64 --layer_type concatsquash --batch_size 20 --lr 0.001 --use_tb --validation 0.1 --uniform_eigval --isotrope_gaussian --beta 50 --mean_of_eigval 0.1
```

#### Graph datasets

QM7:
```
python train.py --dataset qm7 --model moflow --n_flow 32 --n_block 1 --batch_size 100 --lr 0.0004 --noise_scale 0.5674 --use_tb --validation 0.1 --uniform_eigval --beta 157 --mean_of_eigval 0.2140 --n_epoch 1000 --save_each_epoch 1 --isotrope_gaussian
```

<!--QM9:
```
python train.py --dataset qm9 --model moflow --n_flow 32 --n_block 1 --batch_size 100 --lr 0.0006 --noise_scale 0.8251 --use_tb --validation 0.1 --uniform_eigval --beta 133 --mean_of_eigval 0.5040 --n_epoch 1000 --save_each_epoch 1 --isotrope_gaussian
```-->

ESOL:
```
python train.py --dataset esol --model moflow --n_flow 32 --n_block 1 --batch_size 120 --lr 0.0003 --noise_scale 0.2704 --use_tb --validation 0.1 --uniform_eigval --beta 53 --mean_of_eigval 0.9341 --n_epoch 10000 --save_each_epoch 10 --isotrope_gaussian
```

FREESOLV:
```
python train.py --dataset freesolv --model moflow --n_flow 32 --n_block 1 --batch_size 100 --lr 0.0004 --noise_scale 0.5674 --use_tb --validation 0.1 --uniform_eigval --beta 145 --mean_of_eigval 0.8 --n_epoch 10000 --save_each_epoch 10 --isotrope_gaussian
```

## Find Models

If you have disabled evaluation or reduced the size of the evaluation dataset during training, you can compare the 
different checkpoints with each other by calling the get_best_model script:
```
python get_best_model.py --eval_type {evaluation_type} --folder ./checkpoint/{dataset}/{model_type}/{model_name}
```
The previous command has to be used by replacing {dataset},{model_type} and {model_name} by your parameters and 
{evaluation_type} by the evaluation type which can be 'classification', 'regression', or 'projection'.
For example:
```
python get_best_model.py --eval_type classification --folder ./checkpoint/bcancer/seqflow/f32_nfkernel_lmean1_eigvaluniform10_noise01_dimperlab15
```

<!--
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

### Regression

Once models have been trained, in order to choose the best model for regression, the following command has to be used by replacing {dataset},{model_type} and {model_name} by your parameters :
```
python get_best_regression.py --folder ./checkpoint/{dataset}/{model_type}/{model_name}
```
For example:
```
python get_best_regression.py --folder ./checkpoint/swissroll/ffjord/b1_nfkernel_lmean50.0_isotrope_eigvaluniform10_noise01_dimperlab2
```
-->

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

### Regression
```
python evaluate.py --folder ./checkpoint/{dataset}/{model_type}/{model_name} --eval_type regression --model_to_use regression
```
