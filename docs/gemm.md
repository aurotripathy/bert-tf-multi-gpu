### Interpreting the GEMM utilization from the profiler

General Matrix Multiplies (GEMM) exist in the vast majority of deep neural networks today. In the case of BERT it is a major part of the workload. Matrix Multiplies are used to implement fully connected layers and vanilla RNNs.

When performing the GEMM operation C <= aA * B where A, B and C are matrices and either (or both) of A and B can be optionally transposed.


