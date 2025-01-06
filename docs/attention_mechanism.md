# Attention mechanism in Tinychat

## Linear transformation
Input hidden dimension: **[1: sqlen : embed_dim]** (1 is for batch size).

After qkv transformation, the qkv matrices *unshaped* dimensiones: **[1: sqlen: embed_dim]**, since the qkv weight matrices dimensiones are defined as **[embed_dim: embed_dim]**:

```c++
// Note: The "/ 2" comes from the fact the weight is stored in *in4* format, i.e. 1 uint8_t stores 2 weight values.
this->q_proj =
        Linear_FP_int4(Matrix3D<uint8_t>(q_weight, 1, config.embed_dim, config.embed_dim / 2), param_path + "/q_proj");
this->k_proj =
        Linear_FP_int4(Matrix3D<uint8_t>(k_weight, 1, config.embed_dim, config.embed_dim / 2), param_path + "/k_proj");
this->v_proj =
        Linear_FP_int4(Matrix3D<uint8_t>(v_weight, 1, config.embed_dim, config.embed_dim / 2), param_path + "/v_proj");
```
- The matrices so far are *unshaped*, of dimension **[1: sqlen: embed_dim]**. Another function call shapes them into dimension **[num_heads: sqlen: head_dim]**, which represents *Multi-head attention*
```c++
    Matrix3D<float> query_states(query_states_arr, this->num_heads, sqlen, this->head_dim);
    this->shape(query_states_unshape, query_states, sqlen);
```

## Rotary position embedding
```c++
// Rotate position
int start_idx = 0;
if (input.has_past_key_value) start_idx = input.past_key.m_dim_y;
this->rotary_pos_emb.forward(query_states, key_states, start_idx, sqlen);
```
As the code suggests, *Rope* only applies to query and key matrices.

### Rope mechanism

//TODO

## Preparation for attention computation

### Memory buffer

```c++
// Get the memory buffer
float *ret_value_states, *ret_key_states;
if (cache_num[input.layer_idx] == 1) {
	ret_value_states = value_states_arr_cache[input.layer_idx][1];
	ret_key_states = key_states_arr_cache[input.layer_idx][1];
	cache_num[input.layer_idx] = 0;
} else {
	ret_value_states = value_states_arr_cache[input.layer_idx][0];
	ret_key_states = key_states_arr_cache[input.layer_idx][0];
	cache_num[input.layer_idx] = 1;
}
```
The dimension for value_states_arr_cache/key_states_arr_cache is **[num_layers: 2: max_sqlen * embed_dim * sizeof(float)]**. Although the variable name has "cache" in it, nothing is read from it, only written. These 2 spaces act as the memory buffer, storing kv values that are ready to be fed into attention computation. As the dimension suggests, each layer has its own double memory buffer space, used alternately.

```c++
// How value_states_arr_cache/key_states_arr_cache is initialized, for reference
key_states_arr_cache = new float **[config.num_layers];
for (int i = 0; i < config.num_layers; ++i) {
	key_states_arr_cache[i] = new float *[2];
	for (int j = 0; j < 2; ++j) {
		allocate_aligned_memory(key_states_arr_cache[i][j], config.max_sqlen * config.embed_dim * sizeof(float));
	}
}
value_states_arr_cache = new float **[config.num_layers];
for (int i = 0; i < config.num_layers; ++i) {
	value_states_arr_cache[i] = new float *[2];
	for (int j = 0; j < 2; ++j) {
		allocate_aligned_memory(value_states_arr_cache[i][j], config.max_sqlen * config.embed_dim * sizeof(float));
	}
}
```

### Set the kv values right

#### No kv cache / Prefill phase

In prefill phase, the sequence length is more than 1, usually comprised of system prompt and user prompts.

Here the value in value_states_arr is the value matrix generated from the linear transformation,  matrix-matrix multiplications.

The same applies to key matrix. By copying kv values into memory buffers, they are ready for attention computation.

```c++
// Concate with past key, value if exists
int tgz = sqlen;
if (input.has_past_key_value) {
	// # reuse k, v, self_attention
	//...
} else {
// Put the data into the buffer
	memcpy(ret_value_states, value_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(float));
	memcpy(ret_key_states, key_states_arr, (this->num_heads * tgz * this->head_dim) * sizeof(float));
}
Matrix3D<float> final_value_states(ret_value_states, this->num_heads, tgz, this->head_dim);
Matrix3D<float> final_key_states(ret_key_states, this->num_heads, tgz, this->head_dim);
```

#### With kv cache / Decode phase

In decode phase, sequence length is 1, the previously generated token. Thus, kv values are generated from linear transformations that are in fact matrix-vector multiplications. They are not ready to be fed into attention computation unless concatenated with kv cache.

In the implementation, `tgz` goes from 1 to the correct length (being added to the kv cache length).

`past_block` is computed by multiplying the kv cache length and the `head_dim`, while `sq_block` is computed by multiplying `sqlen` (which is 1) and the `head_dim`.

In the iteration, per-headly, kv cache is copied from `input.past_key` and `input.past_value` into `key_ptr` and `val_ptr`, which actually are storing kv cache into the *memory buffer* mentioned above.

Similarly, the newly computed kv values are also copied into memory buffer.

```c++
// Concat with past key, value if exists
int tgz = sqlen;
if (input.has_past_key_value) {
	// reuse k, v, self_attention
	assert(input.past_key.m_dim_z == this->head_dim);
	tgz += input.past_key.m_dim_y;
	float *val_ptr = ret_value_states, *key_ptr = ret_key_states;
	int past_block = input.past_key.m_dim_y * input.past_key.m_dim_z;
	int sq_block = sqlen * this->head_dim; // sqlen is 1
	for (int i = 0; i < input.past_key.m_dim_x; i++) { // for each head...
		// copy the past value
		memcpy(val_ptr, &input.past_value.m_data[past_block * i], past_block * sizeof(float));
		val_ptr += past_block;
		// cancat the new value
		memcpy(val_ptr, &value_states.m_data[sq_block * i], sq_block * sizeof(float));
		val_ptr += sq_block;
		// copy the past key
		memcpy(key_ptr, &input.past_key.m_data[past_block * i], past_block * sizeof(float));
		key_ptr += past_block;
		// cancat the new key
		memcpy(key_ptr, &key_states.m_data[sq_block * i], sq_block * sizeof(float));
		key_ptr += sq_block;
	}
} else {
// Put the data into the buffer
// ...
}
Matrix3D<float> final_value_states(ret_value_states, this->num_heads, tgz, this->head_dim);
Matrix3D<float> final_key_states(ret_key_states, this->num_heads, tgz, this->head_dim);
```

## Attention computation

### Query and Key

The output of qk multiplication is stored in `attn_weights`, of dimension [num_heads: sqlen: tgz].

For example, in prefill phase, let sequence length be 30:

|matrix|dimension|
|------|---------|
|`query_states`|[32: 30: 128]|
|`final_key_states`|[32: 30: 128]|
|`attn_weights`|[32: 30: 30]|

While in decode phase, assuming one new token is generated by the model:

|matrix|dimension|
|------|---------|
|`query_states`|[32: 1: 128]|
|`final_key_states`|[32: 31: 128]|
|`attn_weights`|[32: 1: 31]|

```c++
// QK_BMM
Matrix3D<float> attn_weights(attn_weights_arr, this->num_heads, sqlen, tgz);
this->qk_bmm.forward(query_states, final_key_states, attn_weights);
```

### Causal Mask

The causal mask in decode phase is all 0, with dimension[1: 1: 31] (if in the scenario above).

```c++
// Add mask (enforce causal masking)
batch_Add(attn_weights, input.attention_mask, attn_weights);
// handle numeric instability (by replaceing infinite numbers with smallest representable float)
for (int i = 0; i < attn_weights.length(); i++) {
	if (std::isinf(attn_weights.m_data[i])) {
			attn_weights.m_data[i] = std::numeric_limits<float>::lowest();
	}
}
```

### Softmax and Value

`attn_weights` becomes `attn_probs` after softmax.

```c++
// Softmax QK
// Note: attention weights/scores dimension [num_heads: sqlen: tgz]
Matrix3D<float> attn_probs(attn_weights_arr, this->num_heads, sqlen, tgz);
softmax(attn_weights, attn_probs, 2);
```

### Output

Do note that `input.hidden_states`'s dimension is **[batch_size: sqlen: embed_dim]**, the output matrix should preserve that dimension to ensure consistency.

`attn_output` is computed by multiplying attention probabilities and values states and has dimension **[num_heads: sqlen: head_dim]**, which is not the hidden states dimension, thus the name *untransposed*.

Another function call *unshape* shapes it into the hidden states dimension: **[batch_size: sqlen: embed_dim]**.

The last process is output projection by `o_proj`.

And...everything is done.

```c++
Matrix3D<float> attn_output(attn_output_arr, this->num_heads, sqlen, this->head_dim);
this->pv_bmm.forward_weight_untransposed(attn_probs, final_value_states, attn_output);

Matrix3D<float> attn_output_transpose(attn_output_transpose_arr, 1, sqlen, this->num_heads * this->head_dim);
this->unshape(attn_output, attn_output_transpose, sqlen);

// Output projection
Matrix3D<float> attn_output_fp(attn_output_fp_arr, 1, sqlen, this->num_heads * this->head_dim);
this->o_proj.forward(attn_output_transpose, attn_output_fp);

// Output assignment
output.attn_output = attn_output_fp;
output.past_key_value = {final_key_states, final_value_states};

PROFILE_END(profile_name);
return output;
```
