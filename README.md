# Meta-SAGE: Scale Meta-Learning Scheduled Adaptation with Guided Exploration for Mitigating Scale Shift on Combinatorial Optimization

This repository is the official implementation of **Meta-SAGE: Scale Meta-Learning Scheduled Adaptation with Guided Exploration for Mitigating Scale Shift on Combinatorial Optimization** (ICML 2023). <br>
> https://arxiv.org/abs/2306.02688



![Meta_sage_fig](https://github.com/kaist-silab/meta-sage/assets/71840971/690bd344-2abf-41b4-a7bb-d2e1ab30dcc9)


<br>

## Dependencies

* Python>=3.8
* [PyTorch](http://pytorch.org/)>=1.7

<br>


## Training Scale Meta Learner (SML)


### POMO-TSP


#### 1. Generate data
```
cd ./data_generation  

python generate_data.py --name train --problem tsp --dataset_size 3000 --seed 12345 --graph_size 200 300 400 --data_dir data/train
```
#### 2. Generate target embedding label
```
cd ./train/POMO/TSP/2_Meta

python Generate_label.py --ep 3000 --problem_size 200 --eas_batch_size 50 --eas_num_iter 100 --seed 12345

python Generate_label.py --ep 3000 --problem_size 300 --eas_batch_size 25 --eas_num_iter 100 --seed 12345

python Generate_label.py --ep 3000 --problem_size 400 --eas_batch_size 10 --eas_num_iter 100 --seed 12345
```

#### 3. Training Scale Meta Learner (SML)

```
python SML_train.py --ep 3000 --eas_batch_size 5
```

### POMO-CVRP

#### 1. Generate data
```
cd ./data_generation  

python generate_data.py --name train --problem vrp --dataset_size 3000 --seed 12345 --graph_size 200 300 400 --data_dir data/train
```
#### 2. Generate target embedding label
```
cd ./train/POMO/CVRP/2_Meta

python Generate_label.py --ep 3000 --problem_size 200 --eas_batch_size 50 --eas_num_iter 100 --seed 12345

python Generate_label.py --ep 3000 --problem_size 300 --eas_batch_size 25 --eas_num_iter 100 --seed 12345

python Generate_label.py --ep 3000 --problem_size 400 --eas_batch_size 10 --eas_num_iter 100 --seed 12345
```

#### 3. Training Scale Meta Learner (SML)
```
python SML_train.py --ep 3000 --eas_batch_size 5
```


### Sym_NCO-TSP

#### 1. Generate data
```
cd ./data_generation  

python generate_data.py --name train --problem tsp --dataset_size 3000 --seed 12345 --graph_size 200 300 400 --data_dir data/train
```
#### 2. Generate target embedding label
```
cd ./train/Sym-NCO/Sym-NCO-POMO/TSP/2_Meta

python Generate_label.py --ep 3000 --problem_size 200 --eas_batch_size 50 --eas_num_iter 100 --seed 12345

python Generate_label.py --ep 3000 --problem_size 300 --eas_batch_size 25 --eas_num_iter 100 --seed 12345

python Generate_label.py --ep 3000 --problem_size 400 --eas_batch_size 10 --eas_num_iter 100 --seed 12345
```

#### 3. Training Scale Meta Learner (SML)

```
python SML_train.py --ep 3000 --eas_batch_size 5
```

### Sym_NCO-CVRP

#### 1. Generate data
```
cd ./data_generation  

python generate_data.py --name train --problem vrp --dataset_size 3000 --seed 12345 --graph_size 200 300 400 --data_dir data/train
```
#### 2. Generate target embedding label
```
cd ./train/Sym-NCO/Sym-NCO-POMO/CVRP/2_Meta

python Generate_label.py --ep 3000 --problem_size 200 --eas_batch_size 50 --eas_num_iter 100 --seed 12345

python Generate_label.py --ep 3000 --problem_size 300 --eas_batch_size 25 --eas_num_iter 100 --seed 12345

python Generate_label.py --ep 3000 --problem_size 400 --eas_batch_size 10 --eas_num_iter 100 --seed 12345
```

#### 3. Training Scale Meta Learner (SML)
```
python SML_train.py --ep 3000 --eas_batch_size 5
```


## Testing Meta-SAGE

### POMO-TSP

#### SAGE

```
cd ./test/POMO/TSP/2_SAGE  

python test.py --ep 1000 --problem_size 200 --sage_batch_size 50 --iter 200 --use_bias 

python test.py --ep 128 --problem_size 500 --sage_batch_size 10 --iter 200 --use_bias 

python test.py --ep 128 --problem_size 1000 --sage_batch_size 4 --iter 200 --use_bias 
```



### POMO-CVRP

#### SAGE

```
cd ./test/POMO/CVRP/2_SAGE  

python test.py --ep 1000 --problem_size 200 --sage_batch_size 50 --iter 200 --use_bias 

python test.py --ep 128 --problem_size 500 --sage_batch_size 10 --iter 200 --use_bias 

python test.py --ep 128 --problem_size 1000 --sage_batch_size 4 --iter 200 --use_bias 
```

### Sym_NCO-TSP

#### SAGE

```
cd ./test/Sym-NCO/TSP/2_SAGE  

python test.py --ep 1000 --problem_size 200 --sage_batch_size 50 --iter 200 --use_bias 

python test.py --ep 128 --problem_size 500 --sage_batch_size 10 --iter 200 --use_bias 

python test.py --ep 128 --problem_size 1000 --sage_batch_size 4 --iter 200 --use_bias 
```


### Sym_NCO-CVRP

#### SAGE

```
cd ./test/Sym-NCO/CVRP/2_SAGE  

python test.py --ep 1000 --problem_size 200 --sage_batch_size 50 --iter 200 --use_bias 

python test.py --ep 128 --problem_size 500 --sage_batch_size 10 --iter 200 --use_bias 

python test.py --ep 128 --problem_size 1000 --sage_batch_size 4 --iter 200 --use_bias 
```
