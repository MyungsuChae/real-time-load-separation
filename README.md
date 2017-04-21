This is the first repository of Myungsu Chae.

I'm a newbie in GitHub, so there can be something strange or misused/unused contents. 

The repository will be updated for better understanding as an open source ASAP.

This work is implemented by Myungsu Chae and Sungkwan Jung.

# Overview

Research goal is solving power dis-aggregation problem related to Non-intrusive Load Monitoring. If users can recognize their power consumption of each devices from total power consumption, the users can save their power consumption much easier.

# Dataset

Power consumption dataset of 3 electronic devices which are fan, notebook, light. Those three devices are representatives of motor-based devices, digital devices, heating devices in order.

# Methodology

Device classifier is based on Machine Learning algorithms especially Random Forest and Support Vector Machine (SVM).

# On-going work

There are some critical problems which has to be solved for better saving power consumption.

First, total power consumption can be dis-aggregated to each electronic devices with high performance only if a classifier already has learned from training data composed of power consumption from each devices. But it is impractical that users collect all combination of each devices, so now we are considering Reinforcement Learning concepts for automatic feedback.

Second, power consumption of more than one device is not a linear combination of each devices, so it can be a much difficult problem. Until now in the research, the classifier can separate combination of just two devices with high performance. But it is hard to separate combination of more than two devices, so we are considering Recurrent Neural Network through TensorFlow which is known as good approach in time-series data set.
