import numpy as np
from numpy import linalg
class SVM(object):
    def __init__(self, kernel,kernel_type = 'linear', C=1.0,gamma=0.01, degree=3, max_iter=1000, tol=0.001):
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.support_vector_tol = 0.01
        self.kernel_type = kernel_type
        self.gamma = gamma

    def fit(self, X, y):
        lagrange_multipliers, intercept = self._compute_weights(X, y)
        self.intercept_ = intercept
        support_vector_indices = lagrange_multipliers > self.support_vector_tol
        self.dual_coef_ = lagrange_multipliers[support_vector_indices] * y[support_vector_indices]
        self.support_vectors_ = X[support_vector_indices]

    def _compute_kernel_support_vectors(self, X):
        res = np.zeros((X.shape[0], self.support_vectors_.shape[0]))
        for i,x_i in enumerate(X):
            for j,x_j in enumerate(self.support_vectors_):
                res[i, j] = self.kernel(x_i, x_j)
        return res

    def predict(self, X):
        kernel_support_vectors = self._compute_kernel_support_vectors(X)
        prod = np.multiply(kernel_support_vectors, self.dual_coef_)
        prediction = self.intercept_ + np.sum(prod, 1)
        return np.sign(prediction)

    def score(self, X, y):
        prediction = self.predict(X)
        scores = prediction == y
        return sum(scores) / len(scores)

    def _compute_kernel_matrix_row(self, X, index):
        row = np.zeros(X.shape[0])
        x_i = X[index, :]
        for j,x_j in enumerate(X):
            row[j] = self.kernel(x_i, x_j)
        return row
   
    def _compute_intercept(self, alpha, yg):
        indices = (alpha < self.C) * (alpha > 0)
        return np.mean(yg[indices])
    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)
    def calc_w(self, alpha, y, X):
        return np.dot(X.T, np.multiply(alpha,y))

    def helper(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                    # a : Lagrange multipliers, sv : support vectors.
                    # Hypothesis: sign(sum^S a * y * kernel + b)
                    if self.kernel_type == 'linear':
                        s += a * sv_y * self.linear_kernel(X[i], sv)
                    if self.kernel_type =='gaussian':
                        s += a * sv_y * self.gaussian_kernel(X[i], sv, self.gamma)   # Kernel trick.
                        self.C = None   
                    if self.kernel_type == 'polynomial':
                        s += a * sv_y * self.polynomial_kernel(X[i], sv, self.C, self.degree)
                y_predict[i] = s
            return y_predict + self.b

    def gaussian_kernel(self, x, y, gamma=0.5):
        # Inputs:
        #   x   : input var
        #   y   : support vectors
        #   gamma   : param
        # K(x,xi) = exp(-gamma * sum((x — xi²)).
        return np.exp(-gamma*linalg.norm(x - y) ** 2 )

    def build_Kernelmatrix(self, X, n_samples): #Gram matrix
     K = np.zeros((n_samples, n_samples))
     for i in range(n_samples):
        for j in range(n_samples):
          if self.kernel == 'linear':
            K[i, j] = self.linear_kernel(X[i], X[j])
          if self.kernel=='gaussian':
            K[i, j] = self.gaussian_kernel(X[i], X[j], self.gamma)  
            self.C = None  
          if self.kernel == 'polynomial':
            K[i, j] = self.polynomial_kernel(X[i], X[j], self.C, self.degree)
     return K

    def _compute_weights(self, X, y):
        iteration = 0
        n_samples = X.shape[0]
        alpha = np.zeros(n_samples) # Initialise coefficients to 0  w
        g = np.ones(n_samples) # Initialise gradients to 1
        n_samples, n_features = X.shape #get datasize_shape
        #[Step2]:
        K = self.build_Kernelmatrix(X, n_samples)
        while True:
            yg = g * y

            # Working Set Selection via maximum violating constraints
            indices_y_positive = (y == 1)
            indices_y_negative = (np.ones(n_samples) - indices_y_positive).astype(bool)#(y == -1)
            indices_alpha_upper = (alpha >= self.C)
            indices_alpha_lower = (alpha <= 0)
            
            indices_violate_Bi = (indices_y_positive * indices_alpha_upper) + (indices_y_negative * indices_alpha_lower)
            yg_i = yg.copy()
            yg_i[indices_violate_Bi] = float('-inf') #cannot select violating indices
            indices_violate_Ai = (indices_y_positive * indices_alpha_lower) + (indices_y_negative * indices_alpha_upper)
            yg_j = yg.copy()
            yg_j[indices_violate_Ai] = float('+inf') #cannot select violating indices
            
            i = np.argmax(yg_i)
            j = np.argmin(yg_j)
            self.w = self.calc_w(alpha, y, X)
            self.b = self.calc_b(X, y, self.w)  
            # Stopping criterion: stationary point or maximum iterations
            stop_criterion = yg_i[i] - yg_j[j] < self.tol
            if stop_criterion or (iteration >= self.max_iter and self.max_iter != -1):
                break
            
            # Compute lambda via Newton Method and constraints projection
            lambda_max_1 = (y[i] == 1) * self.C - y[i] * alpha[i]
            lambda_max_2 = y[j] * alpha[j] + (y[j] == -1) * self.C
            lambda_max = np.min([lambda_max_1, lambda_max_2])

            Ki = self._compute_kernel_matrix_row(X, i)
            Kj = self._compute_kernel_matrix_row(X, j)
            lambda_plus = (yg_i[i] - yg_j[j]) / (Ki[i] + Kj[j] - 2 * Ki[j])
            lambda_param = np.max([0, np.min([lambda_max, lambda_plus])])
            
            # Update gradient
            g = g + lambda_param * y * (Kj - Ki)

            # Direction search update
            alpha[i] = alpha[i] + y[i] * lambda_param
            alpha[j] = alpha[j] - y[j] * lambda_param
            
            iteration += 1

        # Compute intercept
        intercept = self._compute_intercept(alpha, yg)
        sv = alpha > 1e-4
        ind = np.arange(len(alpha))[sv]
        self.alpha = alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        self.intercept_ = intercept
        support_vector_indices = alpha > self.support_vector_tol
        self.dual_coef_ = alpha[support_vector_indices] * y[support_vector_indices]
        self.sv = X[support_vector_indices]
        self.sv_y = y[support_vector_indices]
        self.b = 0
        for n in range(len(self.alpha)):
            # For all support vectors:
            self.b += self.sv_y[n]
            self.b -= np.sum(self.alpha * self.sv_y * K[ind[n], sv])
        self.b = self.b / len(self.alpha)

        # Weight vector
        if self.kernel_type == 'linear':
            self.w = np.zeros(n_features)
            for n in range(len(self.alpha)):
                self.w += self.alpha[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None
        return alpha, intercept