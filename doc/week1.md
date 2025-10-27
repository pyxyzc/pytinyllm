#  Impl Scaled_dot_product_attention_simple

Task
* Input: tensors (Q,K,V) have the same dimensions
* Attention Func: Attention = softmax(QK^T/sqrt(d_k)+M)V

Record
* Broadcast mechanism for batch processing
* Check the input param is None or not
* The similarity between query and key - dot product
* How to compute scale? reverse of sqrt of hidden states dim
 
