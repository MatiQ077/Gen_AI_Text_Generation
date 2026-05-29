"""
transformer.py — Decoder-only Transformer model definition (Keras / TensorFlow).

Defines the architecture constants, two custom Keras layers, and a factory
function create_model(). 

Architecture overview:
    Input token IDs (int32, shape [batch, seq_len])
        → TokenAndPositionEmbedding   (learned token + learned positional)
        → N x TransformerBlock        (causal self-attention + FFN + residuals)
        → Dense(vocab_size)           (raw logits, no softmax)

The model is decoder-only: attention uses a causal mask so each position
can only attend to itself and earlier positions.

Logits are returned without softmax so that the training loss can use
from_logits=True (numerically more stable than softmax + cross-entropy).

Key constants (imported by train.py and infer.py — do not redefine elsewhere):
    VOCAB_SIZE = 20000
    MAXLEN     = 256   (TextVectorization output length; model uses MAXLEN-1)
    EMBED_DIM  = 512
    NUM_HEADS  = 2
    FEED_FORWARD_DIM = 512
"""

import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, ops

VOCAB_SIZE       = 20000
MAXLEN           = 256
EMBED_DIM        = 512
NUM_HEADS        = 2
FEED_FORWARD_DIM = 512

#Combine learned token embeddings with learned positional embeddings
class TokenAndPositionEmbedding(layers.Layer):
    
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int):
        super().__init__()
        self.maxlen    = maxlen
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb   = layers.Embedding(input_dim=maxlen,     output_dim=embed_dim)

    def call(self, x):
        seq_len   = ops.shape(x)[-1]
        positions = ops.arange(0, seq_len, 1)
        positions = self.pos_emb(positions)
        x         = self.token_emb(x)
        return x + positions

#Single decoder Transformer block
class TransformerBlock(layers.Layer):
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        key_dim = embed_dim // num_heads
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=rate,
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1   = layers.Dropout(rate)
        self.dropout2   = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attention_output = self.att(
            inputs, inputs,
            use_causal_mask=True,
            training=training,
        )
        attention_output = self.dropout1(attention_output, training=training)
        out1             = self.layernorm1(inputs + attention_output)
        ffn_output       = self.ffn(out1)
        ffn_output       = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Build the whole model
def create_model(
    maxlen: int    = MAXLEN,
    vocab_size: int = VOCAB_SIZE,
    embed_dim: int  = EMBED_DIM,
    num_heads: int  = NUM_HEADS,
    ff_dim: int     = FEED_FORWARD_DIM,
    num_layers: int = 1,
    dropout: float  = 0.1,
) -> keras.Model:
   
    inputs = layers.Input(shape=(maxlen,), dtype="int32")
    x      = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)(inputs)
    for _ in range(num_layers):
        x  = TransformerBlock(embed_dim, num_heads, ff_dim, rate=dropout)(x)
    logits = layers.Dense(vocab_size, name="lm_logits")(x)
    return keras.Model(inputs=inputs, outputs=logits, name="decoder_lm")

if __name__ == "__main__":
    model = create_model(num_layers=1)
    model.summary()
    batch, seq = 2, MAXLEN
    dummy = np.random.randint(0, min(100, VOCAB_SIZE), size=(batch, seq), dtype=np.int32)
    out   = model(dummy, training=False)
    print("dummy input:", dummy.shape, "logits:", out.shape)