
[How to run the code]
1. Important prerequisites:
python 2.7
numpy
scipy
theano
lasagne
nolearn

2. Run the code on validation fold X:
python [method].py X
For example:
python OTM.py 0
python OTM-XE-EMD1.py 1

3. Known bugs
If the number of classes is large (greater than 16), then it will
take a long time to for theano compile the emd_l2() function in emd_loss.py.
This effects OTM-EMD.py. We recommend not to use OTM-EMD.py in this case.


