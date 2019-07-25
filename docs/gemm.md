### Interpreting the GEMM utilization from the profiler

General Matrix Multiplies (GEMM) exist in the vast majority of deep neural networks today. Matrix Multiplies are used to implement fully connected layers and vanilla RNNs. In the case of BERT, it is a major contributor  to the workload. 

When performing the GEMM operation `C = α * A * B + β * C`, where `A`, `B` and `C` are matrices and `α` and `β` are scalars. The A and B matrices can be optionally transposed leading to four variations called  NN, TN, NT and TT kernels respectively where N corresponds to no-transpose and T corresponds to transpose. The TT variation is not used in nenural networks.


C = αAB + βC, C = αA<sup>T</sup>B + βC, C = αAB<sup>T</sup> + βC and C = αA<sup>T</sup> B<sup>T</sup> + βC

```
int i , j , k ; 
for ( i =0; i<M; i ++){ 
  for ( j =0; j<N; j ++){ 
    for ( k=0; k<K; k++){ 
      C[ i ][ j ] += A[ i ][ k] ∗ B[k ][ j ]; 
    } 
}
```
