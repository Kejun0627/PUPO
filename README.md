# PUPO
Official Implementation of paper:  "Purity Law for Neural Routing Problem Solvers with Enhanced Generalizability"\
![image](images/PUPO.png)
[Paper link](https://openreview.net/pdf?id=6KlIzfkTfi)



## ⚙️ Dependencies & Installation

To run this project, please set up the environment as follows:

```bash
# create the environment
conda create -n pupo python=3.11
conda activate pupo
conda install pytorch==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install tensorboard
conda install pandas
conda install scikit-learn
conda install matplotlib
conda install tqdm
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
pip install lkh
```


## ⚙️ Execution


### Model Training

To conduct vanilla train,

```
python train.py --problem TSP/CVRP 
```

To conduct PUPO train,

```
python train_PUPO.py --problem TSP/CVRP
```

### Evaluation 

**Evaluation Dataset**

The randomly generated dataset used in this paper is the same as that in INViT, which is available on [Gdrive address](https://drive.google.com/uc?id=1meYCOULaX_ckosg46Bv1rK8f5sbxMHHV&export=download). The dataset should be downloaded in data/.


We have provided automatic test after every training. If you want to test the performance independently, 

```
python train.py --problem TSP/CVRP --nb_epochs 0 --checkpoint_model={} [--]
```

