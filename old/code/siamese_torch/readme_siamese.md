# Folder content 

 - *create_dataset* notebook : tests on dataset import to pytorch and siamese-ready format. Functions then implemented in *utils.py*
 - *inference* nb : evaluation of the trained networks, on different data **outdated**
 - *inference2* same with some dimension reduction, also **outdated**
 - *siamese_torch* nb : tests on the siamese torch architecures, succesful code is then used in the *siamese_torch.py* script
 - *test_loader* nb : various pytorch training attempts and test, working code used in *siamese_torch.py* 
 ___
 - *siamese_torch2.py* : previous script used for the model training loop.  Consists of the definition of several architectures, data loading and train/eval loop.  
 - *siamese_torch_test.py* : current script used for the model training loop. Several architectures changes ([models.py](code/siamese_torch/siamese_models.py)), data loading and train/eval loop. 
 - *siamese_torch_eval_sizes.py*: Based on current best torch test version. Contains the loop to compare train/eval sizes. Results used in [plots](code/siamese_torch/plots/plots.ipynb)

<!-- Currently need to comment/uncomment models depending on desired pre-trained backbone training,  -->
Improvements of scripts are still planned, this is mostly in development and debug status.  
**usage** : `python3 siamese_torch.py` (requires the right python env)  
Various scripts options are available, to change training hyperparameters and logs.  Training is [wandb](https://wandb.ai/) compatible with the --wanbd arg.
 - **python env :**   
 `conda create --name <envname> python=3.9` and then  
 `pip install -r requirements.txt` (heavy environment, might need some cleaning)
