
# Machine Learning without the pre-image problem thanks to normalizing flow

## Model codes

The implementation of the different models are based on existing codes:
- RealNVP: https://github.com/ikostrikov/pytorch-flows
- FFJORD: https://github.com/rtqichen/ffjord
- Glow: https://github.com/rosinality/glow-pytorch
- MoFlow : https://github.com/calvin-zcx/moflow

## Environment installation

```
conda env create -f environment.yml
conda activate py39
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install chardet
pip install torch_sparse
pip install torch_scatter
pip install numpy==1.23.5
```

## Training

### Paper hyperparameters

Labeled node graph datasets:

| Dataset | Batch size | LR | $\beta$ | Input noise | Mean of $\lambda_i$ ($\sigma^2$ for regression) | $p$ |
|---------|------------|----|---------|-------------|------------|-----|
| QM7 | 150 | 0.00089 | 105 | 0.307 | 0.308 | 0 / 5 |
| ESOL | 150 | 0.00024 | 169 | 0.536 | 0.051 | 0 / 18 |
| FREESOLV | 10 | 0.00020 | 162 | 0.572 | 0.245 | 0 / 6 |
| AIDS | 50 | 0.00008 | 244 | 0.534 | 29 | 0 / 12 |
| MUTAG | 10 | 0.00010 | 249 | 0.322 | 185 | 0 / 12 |

Attributed node graph datasets:

| Dataset | Batch size | LR | $\beta$ | Input noise A | Input noise X | Mean of $\lambda_i$ ($\sigma^2$ for regression) | $p$ |
|---------|------------|----|---------|---------------|---------------|------------|-----|
| Letter-low | 50 | 0.00092 | 241 | 0.391 | 0.185 | 123 | 0 / 13 |
| Letter-med | 250 | 0.0011 | 294 | 0.36 | 0.17 | 24 | 0 / 3 |
| Letter-high | 250 | 0.0015 | 249 | 0.225 | 0.241 | 123 | 0 / 3 |

How to train:
```
python train.py --dataset {dataset} --validation 0.1 --test 0.1 --model {model_type} {model_structure_parameters} {hyperparameters} {additionnal_arguments} {extra}
```
### Details about arguments:

#### {dataset} :  
**Classification :** single_moon, double_moon, iris, bcancer, mnist, MUTAG, Letter-low, Letter-med, Letter-high  
**Regression :** swissroll, diabetes, aquatoxi, fishtoxi, qm7, esol, freesolv  

#### {model_type} :  
**Vector :** seqflow, ffjord  
**Image :** cglow  
**Graphs :** moflow  

#### {model_structure_parameters} : Arguments to modify the structure of the model (optional)  

#### {hyperparameters} :  
**Shared :**  
--batch_size, --lr, --beta, --mean_of_eigval (mean of $\lambda_i$ ($\sigma^2$ for regression)), --add_feature ($p$), --n_epoch  
**Please add the --uniform_eigval argument to define uniformly the eigen values of the covariance matrices.**  
**Non-graph model :**  
--with_noise (noise on input applied during training)
**For graph model (moflow) :**  
--input_scale (Input noise / Input noise A), --input_scale_x (Input noise X to define if attributed node graph)  
**While using regression (not classification) graph datasets please add the --isotrope_gaussian argument.  
In addition, it is recommended to set the --split_graph_dim argument to keep the dimensional split between A and X in the feature space.**  

#### {additionnal_arguments} :  
--save_each_epoch, --save_at_epoch,  
--reduce_train_dataset_size (value between 0 and 1) : reduce the train dataset size while fitting the predictor in the feature space at each validation step.  
--n_permutation_test : how much permutation to use on the train dataset while fitting the predictor in the feature space at each validation step.

#### {extra} :  
--use_tb

<!--

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
ESOL:
```
python train.py --dataset esol --model moflow --n_flow 32 --n_block 1 --batch_size 120 --lr 0.0003 --noise_scale 0.2704 --use_tb --validation 0.1 --uniform_eigval --beta 53 --mean_of_eigval 0.9341 --n_epoch 10000 --save_each_epoch 10 --isotrope_gaussian
```

FREESOLV:
```
python train.py --dataset freesolv --model moflow --n_flow 32 --n_block 1 --batch_size 100 --lr 0.0004 --noise_scale 0.5674 --use_tb --validation 0.1 --uniform_eigval --beta 145 --mean_of_eigval 0.8 --n_epoch 10000 --save_each_epoch 10 --isotrope_gaussian
```
-->

## Find Models

If you have disabled evaluation or reduced the size of the training dataset during validation steps (with --reduce_train_dataset_size), you can compare the 
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

Finally, it is possible to evaluate the models by calling the evaluate.py script, which evaluates the predictor model and generates pre-images by sampling the feature space in different ways.

### Classification
```
python evaluate.py --folder ./checkpoint/{dataset}/{model_type}/{model_name} --eval_type classification --model_to_use classification
```
### Regression
```
python evaluate.py --folder ./checkpoint/{dataset}/{model_type}/{model_name} --eval_type regression --model_to_use regression
```

### Non-graph evaluations :

In addition, other types of evaluation are possible on non-graphical datasets : 
#### Projection (vector and image datasets)
```
python evaluate.py --folder ./checkpoint/{dataset}/{model_type}/{model_name} --eval_type projection --model_to_use projection
```
The evaluation on MNIST for projection can be done on the best classification model by setting model_to_use parameter to classification. 
#### Image datasets generation
```
python evaluate.py --folder ./checkpoint/mnist/cglow/{model_name} --eval_type generation --model_to_use classification
```

