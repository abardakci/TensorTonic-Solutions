import numpy as np

def vit_encoder_block(x: np.ndarray, num_heads: int,
                      Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                      Wo: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 output of one pre-normalized ViT encoder block.
    """
    vit_encoder = ViTEncoder(num_heads, Wq, Wk, Wv, Wo, W1, W2)
    return vit_encoder(x)
    
class ViTEncoder:
    def __init__(self, num_heads: int,
                 Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                 Wo: np.ndarray, W1: np.ndarray, W2: np.ndarray):
        D = Wq.shape[-1] 
        self.ln1 = LayerNorm(D)
        self.ln2 = LayerNorm(D)
        self.attn = MultiHeadSelfAttn(num_heads, Wq, Wk, Wv, Wo)
        self.W1 = W1
        self.W2 = W2
        
    def forward(self, x):
        z = self.ln1(x)
        y = x + self.attn(z)
        z = y + GELU(self.ln2(y) @ self.W1) @ self.W2
        return z

    __call__ = forward
    
class LayerNorm:
    def __init__(self, dims, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(dims)
        self.beta = np.zeros(dims)

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)            
        var = x.var(axis=-1, keepdims=True)
        x_hat = (x - mu) / np.sqrt(var + self.eps)
        return x_hat * self.gamma + self.beta

    __call__ = forward
        
class MultiHeadSelfAttn:
    def __init__(self, num_heads, Wq, Wk, Wv, Wo):
        self.D = Wq.shape[-1]
        self.num_heads = num_heads
        self.dh = self.D // num_heads 
        self.Wq = Wq
        self.Wk = Wk
        self.Wv = Wv
        self.Wo = Wo

    def forward(self, x):
        B, N, D = x.shape
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv
        Q = Q.reshape(B, N, self.num_heads, self.dh).transpose(0,2,1,3)
        K = K.reshape(B, N, self.num_heads, self.dh).transpose(0,2,1,3)
        V = V.reshape(B, N, self.num_heads, self.dh).transpose(0,2,1,3)
        out = softmax((Q @ K.swapaxes(-1, -2)) / np.sqrt(self.dh)) @ V
        out = out.transpose(0,2,1,3).reshape(B, N, D)
        return out @ self.Wo
        
    __call__ = forward

def GELU(x):
    """approximation"""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
