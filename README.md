# Adaptation of TPU-based BERT with TF MirrorredStrategy 

This is a multi-GPU variant of [BERT](https://github.com/google-research/bert) developed by Google. The Google version appears to be written for scaling with TPUs ([tf.contrib.tpu.TPUEstimatorSpec](https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec)). The code runs as-as on a multu-GPU system but seems to fallback to a single GPU (atleast with TF 1.14).


In any case, this variation is a multi-GPU version using the "standard" [tf.estimator.EstimatorSpec](https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec) APIs
