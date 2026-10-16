### Notation

A batch is a matrix $X \in \mathbb{R}^{d \times N}$ with one sample per column. Layer $\ell$ computes

$$
Z^{(\ell)} = W^{(\ell)} A^{(\ell-1)} + b^{(\ell)} \mathbf{1}^\top, \qquad A^{(\ell)} = f\big(Z^{(\ell)}\big), \qquad A^{(0)} = X.
$$

The last layer has no activation: it outputs raw scores (logits) for classification, or predictions for regression.

### Losses

**Regression, mean squared error:**

$$
L = \frac{1}{N}\sum_{i=1}^N \lVert \hat y_i - y_i \rVert^2, \qquad \frac{\partial L}{\partial \hat Y} = \frac{2}{N}(\hat Y - Y).
$$

**Classification, softmax + cross-entropy.** Combining them gives a gradient with a famously clean form:

$$
p = \operatorname{softmax}(z), \qquad L = -\frac{1}{N}\sum_i \log p_{i, y_i}, \qquad \frac{\partial L}{\partial Z} = \frac{1}{N}(P - Y).
$$

It is computed with the log-sum-exp trick, $\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}$ with $m = \max_j z_j$, so huge logits never overflow.

### Backpropagation (the chain rule, layer by layer)

Write $G^{(\ell)} = \partial L / \partial Z^{(\ell)}$. Then

$$
\frac{\partial L}{\partial W^{(\ell)}} = G^{(\ell)} A^{(\ell-1)\top}, \qquad
\frac{\partial L}{\partial b^{(\ell)}} = G^{(\ell)} \mathbf{1}, \qquad
G^{(\ell-1)} = \big(W^{(\ell)\top} G^{(\ell)}\big) \odot f'\big(Z^{(\ell-1)}\big).
$$

The $\frac{1}{N}$ is applied exactly once, in the loss gradient. (The previous version of this project divided by $N$ a second time inside each layer, which silently shrank the learning rate by a factor of $N$.)

### Activation derivatives

| activation | $f(x)$ | $f'(x)$ |
|---|---|---|
| ReLU | $\max(0, x)$ | $\mathbb{1}[x > 0]$ |
| leaky ReLU | $\max(0.01x, x)$ | $1$ if $x>0$, else $0.01$ |
| sigmoid | $\sigma(x) = 1/(1+e^{-x})$ | $\sigma(x)\,(1-\sigma(x))$ |
| tanh | $\tanh x$ | $1 - \tanh^2 x$ |
| GELU | $x\,\Phi(x)$ | $\Phi(x) + x\,\varphi(x)$ |

$\Phi$ and $\varphi$ are the standard normal CDF and density. Sigmoid's derivative is at most $1/4$, so stacking sigmoid layers multiplies gradients by at most $1/4$ per layer: that is the **vanishing gradient** problem you can watch in the gradient-norm chart.

### Why initialisation variance matters

If inputs to a layer have variance $v$ and weights are i.i.d. with variance $s^2$, each pre-activation has variance $n_{\text{in}} s^2 v$. Keeping the signal's scale constant through depth means choosing

- **He:** $s^2 = 2/n_{\text{in}}$. ReLU zeroes half its inputs, so it needs twice the variance.
- **Xavier/Glorot:** $s^2 = 2/(n_{\text{in}} + n_{\text{out}})$, balancing the forward and backward passes for tanh and sigmoid.
- **LeCun:** $s^2 = 1/n_{\text{in}}$.

### Optimisers

$$
\textbf{SGD: } \theta \leftarrow \theta - \eta g \qquad
\textbf{Momentum: } v \leftarrow \beta v + g,\ \theta \leftarrow \theta - \eta v
$$

$$
\textbf{Adam: } m \leftarrow \beta_1 m + (1-\beta_1) g,\quad v \leftarrow \beta_2 v + (1-\beta_2) g^2,\quad
\theta \leftarrow \theta - \eta \frac{m / (1-\beta_1^t)}{\sqrt{v / (1-\beta_2^t)} + \epsilon}
$$

The $(1-\beta^t)$ factors correct the bias from initialising $m$ and $v$ at zero.

### Curvature and step size

Near a minimum, $L(\theta^* + \delta) \approx L(\theta^*) + \tfrac12 \delta^\top H \delta$. Gradient descent multiplies the error along each Hessian eigenvector by $(1 - \eta\lambda_i)$ every step, so it converges only if $|1 - \eta\lambda_i| < 1$ for all $i$, that is

$$
\eta < \frac{2}{\lambda_{\max}}.
$$

The **Loss landscape** tab estimates $\lambda_{\max}$ by power iteration and $\operatorname{tr}(H)$ with Hutchinson's estimator, both from Hessian-vector products alone.

### Linear algebra lab

- **SVD:** $A = U\Sigma V^\top$; the rank-$k$ truncation $A_k$ minimises $\lVert A - B \rVert_F$ over all rank-$k$ matrices $B$ (Eckart–Young), with error $\sqrt{\sum_{i>k}\sigma_i^2}$.
- **PCA:** the eigenvectors of the covariance $C = \frac{1}{N-1} X_c X_c^\top$ are the directions of greatest variance; the eigenvalues are the variances.
- **Fourier transform:** $\hat A = \mathcal{F}(A)$ splits an image into frequencies. Low frequencies carry smooth structure; high frequencies carry edges and noise.
