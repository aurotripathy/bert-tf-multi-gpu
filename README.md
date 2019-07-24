## Adapting TPU-based BERT for multi-GPU scaling with TF MirrorredStrategy 

This is a multi-GPU variant of [BERT](https://github.com/google-research/bert) originally developed by Google. The Google version appears to be written for scaling with TPUs ([tf.contrib.tpu.TPUEstimatorSpec](https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec)). The Google code runs as-is on a multi-GPU system but seems to fallback to a *single* GPU (at least, that's the case with TF 1.14).


In any case, this repo is a variation of the original code using the "standard" [tf.estimator.EstimatorSpec](https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec) APIs to make it scale to multi-GPUs. To get a hands-on intro to the topic of scaling to multiple GPUs, take a look [here](https://github.com/shu-yusa/tensorflow-mirrored-strategy-sample)

Note, you should lean heaviliy of the documentation in the original [BERT](https://github.com/google-research/bert) repo. Documentation provided here is barebones. 

### Requirements
TF 1.14 and Python3

If you are using TF 1.13.1, you may need a compatible `tensorflow-estimator`.

I had to `pip install tensorflow-estimator==1.13.0`

### Multi-GPU Pre-training with Toy Dataset
I'll call this training from scratch. 

You'll need to download and unzip the files at [uncased_L-12_H-768_A-12 model](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip). 

Invoke the script `run_create.sh` to create the TFRecord formatted inputs.

Then invoke the script `run_pre.sh`  to train from scratch. If you want to run the training step (in addition to the eval step), ensure you delete the output folder `/tmp/pretraining_output`.

### Multi-GPU Fine-tuning with Dataset

The BERT dataset format is quite rigid, so I picked up a reasonable dataset for fine-tuning on BERT from [here](https://github.com/craic/bert_paper_classification). 

Download the `dev.tsv`, `test.tsv`, `train.tsv` and deposit them in the `data` folder. 

Pre-trained BERT comes in two sizes - Base and Large - we're using the base model.

Run the script `run_classify.sh`

### Profiling BERT on Multi GPU

The `nvprof` profiler reveals that a majority of time is spent primarily in `sgemm` (49.32%) and `ncclAllReduce` (17.49%) kernels. 
Other kernels excercised are, scalar product (10.28%) and  memcpy P2P/D2D (5.31%), 

<p float="left">
  <img src="/docs/nvprof.PNG" width="2000" />
</p>



