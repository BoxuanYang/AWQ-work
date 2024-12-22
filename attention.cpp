#include <iostream>
#include <cmath>  

/*
naive self attention -- single head

Shape:
  1. Q: N by d
  2. K: N by d
  3. V: N by d
*/
void naive_self_attention(float *output, float *Q, float *K, float *V, int N, int d){

    // 1. S = Q * K^T, N by N
    float *S = (float *) malloc(N * N * sizeof(float));
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            // S[i][j] = Q[i][k] * K^T[k][j] = Q[i][k] * K[j][k]
            float val = 0.0f;
            for(int k = 0; k < d; k++){
                val += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j]  = val;
        }
    }
    // 2. S = softmax(S)
    naive_softmax(S, N, N);

    // 3. output = S @ V
    matmul(output, S, V, N, N, d);

    free(S);
}


/*
output = X Y

Shape:
  X:      m by k
  Y:      k by n
  output: m by n
*/
void matmul(float *output, float *X, float *Y, int m, int k, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            float val = 0;
            // output[i][j] += X[i][kk] * Y[kk][j]
            for(int kk = 0; kk < k; kk++){
                val += X[i * k + kk] * Y[kk * n + j];
            }
            output[i * n + j] = val;
        }
    }
    return;
}

/*
Perform safe softmax.

1. Compute max_elem
2. Compute sum of exp(xi - max_elem), i.e., summ
3. Compute exp(xi - max_elem)

Shape:
  input: m by n
*/
void naive_softmax(float *input, int m, int n){
    // iterate over all rows
    for(int i = 0; i < m; i++){
        naive_softmax_vector(&input[i * n], n);
    }
    return;
}

/*
Perform safe softmax of a vector

1. Compute max_elem
2. Compute sum of exp(xi - max_elem), i.e., summ
3. Compute exp(xi - max_elem)

Shape:
  intput: m
*/
void naive_softmax_vector(float *input, int m){
    float max_elem = 1.0f;

    // 1. Compute max_elem
    for(int i = 0; i < m; i++){
        if(input[i] > max_elem){
            max_elem = input[i];
        }
    }

    // 2. Compute sum of exp(xi - max_elem), i.e., summ
    float summ = 0.0f;
    for(int i = 0; i < m; i++){
        summ += exp(input[i] - max_elem);
    }

    // 3. Compute exp(xi - max_elem) / summ
    for(int i = 0; i < m; i++){
        input[i] = exp(input[i] - max_elem) / summ;
    }
    return;
}

/*
Shape:
  input:  m by n
  weight: m by n
*/
void rms_norm(float *output, float *input, float *weight, int m, int n){
    for(int i = 0; i < m; i++){
        rms_norm_vector(&output[i * n], &input[i * n], &weight[i * n], n);
    }
    return;
}

/*

Calculate:
  input[i] = inpupt[i] * weight[i] / rms
where,
  rms = 1 / sqrt(1/n * sum(xi * xi) + 1e-5f)
Shape:
  input:  n
  weight: n
  output: n
*/
void rms_norm_vector(float *output, float *input, float *weight, int n){
    // calculate sum of squares
    float summ = 0.0f;
    for (int j = 0; j < n; j++) {
        summ += input[j] * input[j];
    }
    summ /= n;
    summ += 1e-5f;
    summ = 1.0f / sqrtf(summ);

    // normalize and scale
    for (int j = 0; j < n; j++) {
        output[j] = weight[j] * (summ * input[j]);
    }
}

/*
TODO:
  Finish this function
Compute SwiGLU:
  SwiGLU(x) = Swish(W1x+b) ⊗ (Vx+c)

Shape:
  W1:    m by n
  V:     m by n
  input: n
  c:     m
  b:     m
*/
void swiglu(float *output, 
            float *input, 
            float *W1, float *b, 
            float *V, float *c,
            float *beta,
            int m, int n){

    float *x_w1_plus_b = (float *) malloc(m * sizeof(float));

    // W1x + b
    matmul(x_w1_plus_b, W1, input, m, n, 1)

    return;
}


/*
Compute swish:
  Swish(x)  = x * sigmoid(ßx)

Shape:
  input:  n
  beta:   n
  output: n
*/
void swish(float *output, float *input, float *beta, int n){
    for(int i = 0; i < n; i++){
        float val = input[i] * beta[i];
        
        val = 1.0f / (1 + exp(-val));

        val = val * input[i];

        output[i] = val;
    }
    return;
}