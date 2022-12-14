# Consistency is the key to further mitigating catastrophic forgettin in continual learning 
Official Repository for for CoLLAs 2022 paper "[Consistency is the key to further mitigating catastrophic forgettin in continual learning](https://arxiv.org/abs/2207.04998)"

This repo is built on top of the [Mammoth](https://github.com/aimagelab/mammoth) continual learning framework

## Setup

+ Use `python main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters for each of the evaluation setting from the paper.
+ To reproduce the results in the paper run the following

    `python main.py --seed <seed> --dataset <dataset> --img_size <size> --model <model> --buffer_size <buffer_size> --load_best_args --pretext_task <regularizer>`

 Examples:
    
    python main.py --seed 10 --dataset seq-cifar10 --img_size 32 --model cr  --buffer_size 200 --load_best_args --pretext_task mse
  
    python main.py --seed 10 --dataset seq-cifar100 --img_size 32 --model cr  --buffer_size 500 --load_best_args --pretext_task linf

    python main.py --seed 10 --dataset seq-tinyimg --img_size 64 --model cr  --buffer_size 5120 --load_best_args --pretext_task l1
    

## Requirements

- torch==1.7.0

- torchvision==0.9.0 

- quadprog==0.1.7

## Cite Our Work

If you find the code useful in your research, please consider citing our paper:


    @InProceedings{pmlr-v199-bhat22b,
      title = 	 {Consistency is the Key to Further Mitigating Catastrophic Forgetting in Continual Learning},
      author =       {Bhat, Prashant Shivaram and Zonooz, Bahram and Arani, Elahe},
      booktitle = 	 {Proceedings of The 1st Conference on Lifelong Learning Agents},
      pages = 	 {1195--1212},
      year = 	 {2022},
      editor = 	 {Chandar, Sarath and Pascanu, Razvan and Precup, Doina},
      volume = 	 {199},
      series = 	 {Proceedings of Machine Learning Research},
      month = 	 {22--24 Aug},
      publisher =    {PMLR},
      pdf = 	 {https://proceedings.mlr.press/v199/bhat22b/bhat22b.pdf},
      url = 	 {https://proceedings.mlr.press/v199/bhat22b.html},
    }
