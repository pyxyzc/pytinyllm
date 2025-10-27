import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    
    # Input Shape
    print(f"query shape: {query.shape}")
    print(f"key shape: {key.shape}")
    print(f"value shape: {value.shape}")

    if scale is None:
        print(f"scale is None")
        scale = mx.rsqrt(query.shape[-1])
        print(f"After calculation, scale value: {scale}")
    else:
        print(f"scale value: {scale}")
    if mask is None:
        print("mask is None")
    else:
        print(f"mask shape: {mask.shape}")
    
    scores = mx.multiply(mx.matmul(query, key.swapaxes(-2, -1)), scale)
    if mask is not None:
        scores = scores + mask
    return mx.matmul(mx.softmax(scores, -1), value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        pass

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        pass


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
