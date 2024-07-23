# DNN Training & Architectures

*In artificially intelligent systems, does training neural networks with different neuroscience-inspired algorithms impact continual learning capabilities?*


This project is testing the continual learning abilities of new developments in biologically-plausible alternatives to back propagation in deep learning and different deep learning architectures. 

The folder **CL** is at its beginning phases and is currently testing the continual learning abilities of equilibrium propagation (https://github.com/smonsays/equilibrium-propagation), fixed-weight difference target propagation (https://doi.org/10.48550/arXiv.2212.10352), difference target propagation (https://doi.org/10.48550/arXiv.1412.7525), predictive coding (https://github.com/BerenMillidge/PredictiveCodingBackprop/blob/master/cnn.py),and back propagation neural networks when trained sequentially in the supervised learning setting, specifically on different multi-class classification tasks.

Also in the **CL** folder are different architectures (also with different training methods - those of their own method) including Kolmogorov-Arnold Network (KAN) (adapted from https://www.kaggle.com/code/mickaelfaust/99-kolmogorov-arnold-network-conv2d-and-mnist?scriptVersionId=177633280) and a Hamiltonian Bitwise Whole-part Architecture (Hnet) (adapted from https://github.com/DartmouthGrangerLab/hnet), to complement the more traditional CNN architectures used to test the alternative training methods listed previously.



