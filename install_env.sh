conda env create -f envronement.yml
conda activate py39
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install chardet
pip install torch_sparse
pip install torch_scatter
pip install numpy==1.23.5
