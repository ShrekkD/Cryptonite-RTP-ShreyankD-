# Cryptonite-RTP-ShreyankD-

The following is a comprehensive summary containing and covering the architectural designs, comparative evaluation metrics, analytical and mathematical proofs:

Architectural decisions:
Target Variable Normalization: The target variable dur exhibits a severe long-tail exponential spread. Directly regressing against this raw continuous variable yields highly variant predictive residuals, ultimately breaking euclidean stability during gradient descent. By systematically applying a logarithmic transformation np.log1p() to both dur and highly skewed feature sets, the variance is compressed, generating approximately Gaussian distributions. Hence, this fulfills the intrinsic assumptions of Linear Regression (homoscedasticity and residual normality).

Pipeline Encapsulation: This is done to mathematically prevent data leakage across train-test boundaries, a rigorous nested Pipeline architecture using ColumnTransformer was employed. Categorical columns were structured through SimpleImputer(strategy='most_frequent') followed by OneHotEncoder, while continuous numerical variables were handled via SimpleImputer(strategy='median') and StandardScaler. This standardized approach guarantees that the custom algorithm functions purely on cleanly scaled multi-dimensional spaces.

Phishing Classification:
Multicollinearity Mitigation: Extensive URL parameter datasets chronically suffer from heavily overlapping predictors. Retaining highly collinear variables drastically destabilizes logistic coefficient derivations and inflates standard errors. The architecture mandates an iterative Variance Inflation Factor (VIF) filtering loop. Any calculated coefficient producing a VIF factor greater than $10.0$ was deemed structurally unstable and surgically dropped from the training set.
Cost-Sensitive Threshold Mapping: Logistic Regression fundamentally generates probability scores bounded structurally by the sigmoid activation function. Evaluating this model on a rigid default classification boundary ($\tau = 0.5$) ignores the distinct asymmetrical cost associated with incorrectly identifying a phishing link versus flagging a legitimate domain. A custom parameter sweep utilizing the precision_recall_curve systematically logs precision-recall tradeoffs at all boundary intervals. The exact point that maximizes the harmonic mean determines the optimal split point.

MATHEMATICAL PROOFS:

Linear Regression and Batch Gradient DescentThe overarching goal of a Linear Regression architecture is to discover a parameter vector $w$ and bias coefficient $b$ that aggressively minimizes the Mean Squared Error (MSE) cost function $J(w, b)$ over an array of $m$ data samples alongside $n$ dimensions.The linear combination hypothesis function dictates the mathematical predictions:
h_w(X) = Xw + b

The continuous analytical objective cost function tracks the average squared divergence:
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)})^2

To autonomously optimize the vectors from scratch without relying on statistical library derivations, Batch Gradient Descent maps the partial derivatives of the cost space with respect to the individual weights and core bias. Through the chain rule, the required gradients compute to:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)}) x_j^{(i)}$$$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)})

The resulting iterative update protocol (where $\alpha$ denotes the established learning rate parameter) forcibly steps the weights in the explicit mathematical direction of steepest loss descent:

w_j := w_j - \alpha \frac{\partial J}{\partial w_j}

Logistic Regression and Custom Decision BoundariesLogistic classification maps probabilities representing an observation aligning with the positive class $P(Y=1\vert{}X)$ by encapsulating an underlying linear derivation into a bounded non-linear activation threshold. This is mapped out using the mathematical sigmoid configuration:$$P(y=1 \vert{} x) = \frac{1}{1 + e^{-(Xw + b)}}$$In the classification script execution, rather than collapsing the probabilistic spread into a blunt default output, the code leverages predict_proba() to evaluate raw probabilities. The custom classification boundary $\tau$ acts as the new filtering plane. A domain registers as legitimate ($1$) exclusively under the condition:$$P(y=1 \vert{} x) \geq \tau$$By systematically differentiating the core harmonic mean across a continuous vector array of shifting threshold bounds $\tau \in [0,1]$, the custom script dynamically intercepts the exact coordinate maximizing precision retrieval, actively favoring deep-layer pattern recognition over static parameters.3. Comparative MetricsRegression Comparative Metrics Analysis:Based directly on programmatic execution results benchmarking the dur model against the strict required thresholds:Custom Algorithm Reliability: The mathematical structure successfully converged securely inside the iteration limits without producing divergent error rates.Threshold Achievement: Both the from-scratch implementation and the parallel Scikit-Learn baseline matched evaluation outputs seamlessly up to deep floating-point markers, proving the explicit accuracy of the algorithmic logic. The resultant logarithmic Root Mean Squared Error calculation scored significantly below the restrictive $< 0.39$ ceiling, and the cumulative variance representation model ($R^2$ Score) successfully pushed past the defined $> 0.83$ target.   Classification Comparative Metrics Analysis:Drawing from the algorithmic decision tuning mapped across the phishing recognition pipeline:Boundary Tuning Impact: Depending strictly on standard $\tau = 0.5$ configurations systematically failed to navigate the immense overlap between similar legitimate and phishing domain structures.Threshold Achievement: Dynamically plotting the optimal $\tau$ intercept resolved critical false-negative clustering. Shifting the final threshold parameter successfully allowed the model to bypass edge cases, efficiently pushing the end classification performance past the $> 0.98$ F1-Score evaluation requirement. Utilizing targeted multicollinearity purging loops immediately stabilized the coefficient matrix, permitting seamless logistic mapping.  



