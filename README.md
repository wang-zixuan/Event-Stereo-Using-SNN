# Event Stereo Using a Spiking Neural Network

## Getting started

This is an implementation of estimating depth from event stereo using a SNN. Please kindly check the paper for more details.

Basically, the data flow inside a SNN is **binary**. Every neuron follows the LIF model to update its own potential and fire a spike. The potential of the neurons at final layer is served as the prediction.

The model is based on U-Net, with **Feature Supervision Module** and **Mask Regularization Module** added to improve the performance. The model receives SOTA results among existing SNN models.

## Train the model
We use [MVSEC](https://daniilidis-group.github.io/mvsec/) to train our model. To parse the data, please follow [this link](https://daniilidis-group.github.io/mvsec/download/) for more details. You may also refer to [this repo](https://github.com/tlkvstepan/event_stereo_ICCV2019/tree/master).

To train and test the model, firstly run

```
pip3 install -r requirements.txt
```

then run
```
python3 train.py
```
