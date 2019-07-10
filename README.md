# Adaptation of TPU-based BERT with TF MirrorredStrategy 

This is a multi-GPU variant of [BERT](https://github.com/google-research/bert) developed by Google. The Google version appears to be written for scaling with TPUs ([tf.contrib.tpu.TPUEstimatorSpec](https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec)). The Google code runs as-is on a multi-GPU system but seems to fallback to a single GPU (at least thats the casw with TF 1.14).


In any case, this repo is a variation of the original code using the "standard" [tf.estimator.EstimatorSpec](https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec) APIs to make it scale to multi-GPUs 

### Multi-GPU Pre-training with Toy Dataset
I'll call this training from scratch. 

You'll need to download and unzip the files at [uncased_L-12_H-768_A-12 model](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip).

Invoke the script `run_create.sh` to create the tfrecord formatted inputs.

Then invoke the script `run_pre.sh`

### Multi-GPU Fine-training with Dataset


