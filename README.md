# partial_neural_collapse

This repo is based on the GitHub repository provided by Rupert Wu [here](https://github.com/rhubarbwu/neural-collapse) published under the MIT license. 


The ViT implementation in vit.py is taken from https://github.com/tintn/vision-transformer-from-scratch/tree/main, which was published under the MIT license.
The VGG implementation in vgg.py is taken from https://github.com/jerett/PyTorch-CIFAR10. The author granted explicit permission to use the code.

To run an experiment, specify the parameters in the config file and call the following run. 

python main.py --config-name config.yaml --multirun hydra/launcher=joblib hydra.launcher.n_jobs=1 