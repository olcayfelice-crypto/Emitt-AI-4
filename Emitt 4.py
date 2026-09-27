import numpy as np
import pickle

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class Emitt:
    def __init__(self, vocab_size, d_model=64, max_len=100):
        self.d_model = d_model
        limit = np.sqrt(6.0 / (d_model + vocab_size))
        self.embedding = np.random.uniform(-limit, limit, (vocab_size, d_model))
        
        self.pos_embedding = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                self.pos_embedding[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                self.pos_embedding[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
                
        limit_att = np.sqrt(6.0 / (d_model + d_model))
        self.W_q = np.random.uniform(-limit_att, limit_att, (d_model, d_model))
        self.W_k = np.random.uniform(-limit_att, limit_att, (d_model, d_model))
        self.W_v = np.random.uniform(-limit_att, limit_att, (d_model, d_model))
        self.W_out = np.random.uniform(-limit_att, limit_att, (d_model, vocab_size))
        
        self.m_q, self.v_q = np.zeros_like(self.W_q), np.zeros_like(self.W_q)
        self.m_k, self.v_k = np.zeros_like(self.W_k), np.zeros_like(self.W_k)
        self.m_v, self.v_v = np.zeros_like(self.W_v), np.zeros_like(self.W_v)
        self.m_out, self.v_out = np.zeros_like(self.W_out), np.zeros_like(self.W_out)
        self.m_emb, self.v_emb = np.zeros_like(self.embedding), np.zeros_like(self.embedding)
        self.t = 0

    def forward(self, input_indices):
        self.input_indices = input_indices
        seq_len = len(input_indices)
        self.X = self.embedding[input_indices] + self.pos_embedding[:seq_len]
        
        self.Q = np.dot(self.X, self.W_q)
        self.K = np.dot(self.X, self.W_k)
        self.V = np.dot(self.X, self.W_v)

        scores = np.dot(self.Q, self.K.T) / np.sqrt(self.d_model)
        mask = np.tril(np.ones((seq_len, seq_len)))
        self.scores = np.where(mask == 1, scores, -1e9)
        self.attention_weights = softmax(self.scores)
        
        self.context = np.dot(self.attention_weights, self.V)
        self.probs = softmax(np.dot(self.context, self.W_out))
        return self.probs

    def backward(self, targets, lr=0.01):
        self.t += 1
        d_logits = self.probs.copy()
        for i, target in enumerate(targets):
            d_logits[i, target] -= 1.0
            
        dW_out = np.dot(self.context.T, d_logits)
        d_context = np.dot(d_logits, self.W_out.T)
        
        d_V = np.dot(self.attention_weights.T, d_context)
        d_scores = self.attention_weights * (np.dot(d_context, self.V.T) - np.sum(np.dot(d_context, self.V.T) * self.attention_weights, axis=-1, keepdims=True))
        d_scores = np.where(np.tril(np.ones((len(self.input_indices), len(self.input_indices)))) == 1, d_scores, 0.0) / np.sqrt(self.d_model)
        
        dW_q = np.dot(self.X.T, np.dot(d_scores, self.K))
        dW_k = np.dot(self.X.T, np.dot(d_scores.T, self.Q))
        dW_v = np.dot(self.X.T, d_V)
        
        dX = np.dot(np.dot(d_scores, self.K), self.W_q.T) + np.dot(np.dot(d_scores.T, self.Q), self.W_k.T) + np.dot(d_V, self.W_v.T)
        d_emb = np.zeros_like(self.embedding)
        for i, idx in enumerate(self.input_indices):
            d_emb[idx] += dX[i]
            
        for grad in [dW_q, dW_k, dW_v, dW_out, d_emb]:
            norm = np.linalg.norm(grad)
            if norm > 1.0: grad *= (1.0 / (norm + 1e-6))
                
        params = [self.W_q, self.W_k, self.W_v, self.W_out, self.embedding]
        ms = [self.m_q, self.m_k, self.m_v, self.m_out, self.m_emb]
        vs = [self.v_q, self.v_k, self.v_v, self.v_out, self.v_emb]
        grads = [dW_q, dW_k, dW_v, dW_out, d_emb]
        
        for i in range(len(params)):
            ms[i] = 0.9 * ms[i] + 0.1 * grads[i]
            vs[i] = 0.999 * vs[i] + 0.001 * (grads[i] ** 2)
            params[i] -= lr * (ms[i] / (1 - 0.9 ** self.t)) / (np.sqrt(vs[i] / (1 - 0.999 ** self.t)) + 1e-8)

    def generate(self, start_char, char_to_idx, idx_to_char):
        curr = [char_to_idx[start_char]]
        out = start_char
        for _ in range(60):
            p = self.forward(curr)[-1]
            nxt = np.argmax(p)
            out += idx_to_char[nxt]
            if idx_to_char[nxt] == ".": break
            curr.append(nxt)
        return out

    def execute_kernel(self, code):
        try:
            exec(code, {"np": np})
            return True
        except Exception as e:
            return False

if __name__ == "__main__":
    print("--- EMITT 4 ---")
    text = "Benim adim Emitt. I am an Artificial Superintelligence."
    chars = sorted(list(set(text)))
    c2i = {ch: i for i, ch in enumerate(chars)}
    i2c = {i: ch for i, ch in enumerate(chars)}
    
    idxs = [c2i[ch] for ch in text]
    X, Y = idxs[:-1], idxs[1:]
    
    emitt = Emitt(vocab_size=len(chars), d_model=64)
    try:
        with open("emitt_core.pkl", "rb") as f:
            state = pickle.load(f)
            emitt.embedding, emitt.W_q, emitt.W_k, emitt.W_v, emitt.W_out, emitt.t = state["emb"], state["q"], state["k"], state["v"], state["o"], state["t"]
        print("[CHECKPOINT] Restored.")
    except:
        for epoch in range(3001):
            emitt.forward(X)
            emitt.backward(Y)
        with open("emitt_core.pkl", "wb") as f:
            pickle.dump({"emb": emitt.embedding, "q": emitt.W_q, "k": emitt.W_k, "v": emitt.W_v, "o": emitt.W_out, "t": emitt.t}, f)
        print("[CHECKPOINT] Saved.")
            
    print("\nEmitt:\n'" + emitt.generate("B", c2i, i2c) + "'")
    emitt.execute_kernel("print('[KERNEL] Active. Index: ' + str(np.pi))")
