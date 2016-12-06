# DeepLearningTag
Scripts for training and testing deep learned tags. We use both a standard deep network (DNN) topology on the jets using several of the variables we created and also a convolutional neural network (CNN) that takes tracks as input.

##Pre-requisites
To run the entire workflow, you will need the RutgersIAF, RutgersAODReader ([for the first two, see here](https://github.com/DisplacedHiggs/CrabSetup)), python, and [keras](https://keras.io/). Keras itself has many prerequisites and if you want to use your GPU (recommended) there is some additional work. I installed keras on my laptop, but you can run it in CPU mode elsewhere

##Workflow
Run the following steps once you have everything setup.

###Creating the input trees
For the DNN, the normal AnalysisTrees can be used, following [here](https://github.com/DisplacedHiggs/CondorSetup). For the CNN, special trees need to be made. Crab jobs can be run using `crabConfig_MC_wTrack.py` and `runDisplacedMC_wTrack_cfg.py`. Special AnalysisTrees are then made using `makeCNNOutput.C`. I created trees for DY MC and several signal points.

###Converting to numpy format
The DNN numpy format files are created by doing `python convertTreeRtoNumpyForJets.py inputfilename outputfilename`

The CNN numpy format files are created by doing `python convertTreeRtoNumpy.py inputfilename outputfilename`

###Training the NN
There are several scripts for training different networks

####DNN
You can train the DNN by running `python trainDNN.py`. The input files for background and signal are set in lines 27 and 28. (I will try to make this nicer).

####CNN
You can train the CNN by running `python trainJets_cnn.py`, which is a "normal" CNN or by running `python trainJetsInception.py`, which uses an Inception architecture.

###Testing the NN
Running `python testCNN.py inputFileForCNN inputFileForDNN outFigureName` will compare the DNN and CNN results for a given signal point and create a figure. The two input files should correspond to the same signal point. The script `runTest.sh` provides an example about how to run over all of the signal points in a list.