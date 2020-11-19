import numpy as np
import numpy.linalg as lin
import scipy.optimize as opt
import copy

# lib for high-precision math
import mpmath as mp; mp.mp.dps = 50
def mp_elementwise_div(a, b):
    if not (a.rows == b.rows and a.cols == b.cols):
        raise ValueError("incompatible dimensions for elementwise division")
    new = a.ctx.matrix(a.rows, a.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            new[i,j] = a[i,j] / b[i,j]
    return new
def mp_elementwise_mul(a, b):
    if not (a.rows == b.rows and a.cols == b.cols):
        raise ValueError("incompatible dimensions for elementwise division")
    new = a.ctx.matrix(a.rows, a.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            new[i,j] = a[i,j] * b[i,j]
    return new
def float2mpf_1D(array):
    return np.array([mp.mpf(x) for x in array])
        


# Class for gaussian processes
class GP:
    def __init__(self, kernel, hypers_init, hypers_scale=None, hypers_prior_mu=None, hypers_prior_sigma=None, 
                       kernel_grad=None, sigma_noise=None, verbose=0):
        self.kernel = kernel
        self.kernel_grad = kernel_grad
        if kernel_grad is not None:
            self._saved_hypers = None
            self._saved_covmat = None
            self._saved_covmatinv = None
            self._saved_alpha = None
        
        self.hypers_init = float2mpf_1D(hypers_init)
        self.hypers = float2mpf_1D(hypers_init)
        self.hypers_scale = float2mpf_1D(hypers_scale) if hypers_scale is not None else float2mpf_1D(np.ones(len(hypers_init)))
        self.hypers_scaled = self.hypers * self.hypers_scale
        
        self.hypers_prior_mu = hypers_prior_mu
        self.hypers_prior_sigma = hypers_prior_sigma
        
        self.sigma_noise = sigma_noise if sigma_noise is not None else 0
        
        self.X = None
        self.y = None
        self.dy = None
    
        self.curr_err = None
        
        #verbose: 0-nothing, 1-likelihood, 2-likelihood+grad, 3-grad, 4-grad+check_grad
        self.verbose = verbose
        
    def kernel_mat(self, X1, X2, hypers=None, **kwargs):
        X1 = X1 if isinstance(X1[0], (list, tuple)) else [X1]
        X2 = X2 if isinstance(X2[0], (list, tuple)) else [X2]
        if hypers is None:
            hypers = self.hypers_scaled
        z = mp.matrix([[self.kernel(x1, x2, hypers, **kwargs) for x2 in X2] for x1 in X1])
        return z

    def minus_loglikelihood(self, hypers):
        hypers = float2mpf_1D(hypers) * self.hypers_scale
        if (self.hypers_prior_mu is not None) and (self.hypers_prior_sigma is not None):
            prior = mp.fsum(((hypers - self.hypers_prior_mu) / self.hypers_prior_sigma)**2)
        else:
            prior = 0

        K = self.kernel_mat(self.X, self.X, hypers)
        cov_mat = K + mp.eye(K.rows)*self.sigma_noise**2
        if self.dy is not None:
            cov_mat += mp.diag(self.dy)**2
        alpha = mp.cholesky_solve(cov_mat, mp.matrix(self.y))

        loglikelihood = - prior - (mp.matrix(self.y).T*alpha)[0] - mp.log(mp.det(cov_mat))
        if self.verbose >= 1:
            print(hypers / self.hypers_scale, -loglikelihood)
        return -loglikelihood

    def minus_loglikelihood2(self, hypers):
        hypers = float2mpf_1D(hypers) * self.hypers_scale
        if (self.hypers_prior_mu is not None) and (self.hypers_prior_sigma is not None):
            prior = mp.fsum(((hypers - self.hypers_prior_mu) / self.hypers_prior_sigma)**2)
        else:
            prior = 0

        if np.all(self._saved_hypers == hypers) and (len(self.y) == self._saved_alpha.rows):
            alpha = self._saved_alpha
            cov_mat = self._saved_covmat
        else:
            #print("calculate all")
            K = self.kernel_mat(self.X, self.X, hypers)
            cov_mat = K + mp.eye(K.rows)*self.sigma_noise**2
            if self.dy is not None:
                cov_mat += mp.diag(self.dy)**2
            cov_mat_inv = cov_mat**(-1)
            alpha = cov_mat_inv * mp.matrix(self.y)

            self._saved_hypers = hypers
            self._saved_covmat = cov_mat
            self._saved_covmatinv = cov_mat_inv
            self._saved_alpha = alpha

        loglikelihood = - prior - (mp.matrix(self.y).T*alpha)[0] - mp.log(mp.det(cov_mat)) 
        if self.verbose in [1, 2]:
            print(hypers / self.hypers_scale, -loglikelihood, sep="\t")
        return -loglikelihood

    def grad_minus_loglikelihood2(self, hypers):
        hypers = float2mpf_1D(hypers) * self.hypers_scale
        if (self.hypers_prior_mu is not None) and (self.hypers_prior_sigma is not None):
            prior_der = 2 * (hypers - self.hypers_prior_mu) / (self.hypers_prior_sigma**2)
        else:
            prior_der = 0
        
        K_ders = [mp.matrix(len(self.X)) for _ in range(len(hypers))]
        for i in range(len(self.X)):
            for j in range(len(self.X)):
                ders = self.kernel_grad(self.X[i], self.X[j], hypers)
                for k in range(len(hypers)):
                    K_ders[k][i, j] = ders[k]

        if np.all(hypers == self._saved_hypers) and (len(self.y) == self._saved_alpha.rows):
            alpha = self._saved_alpha
            cov_mat_inv = self._saved_covmatinv
        else:
            #print("calculate all in Grad")
            K = self.kernel_mat(self.X, self.X, hypers)
            cov_mat = K + mp.eye(len(self.X))*self.sigma_noise**2
            if self.dy is not None:
                cov_mat += mp.diag(self.dy)**2
            cov_mat_inv = cov_mat**(-1)
            alpha = cov_mat_inv * mp.matrix(self.y)

            self._saved_hypers = hypers
            self._saved_covmat = cov_mat
            self._saved_covmatinv = cov_mat_inv
            self._saved_alpha = alpha

        m1 = alpha * alpha.T - cov_mat_inv
        grad_loglikelihood = - prior_der + np.array([float(mp.fsum(mp_elementwise_mul(m1, K_ders[i].T))) 
                                                     for i in range(len(hypers))])         
        grad_loglikelihood = grad_loglikelihood * self.hypers_scale
        grad_loglikelihood = np.array([np.float64(x) for x in grad_loglikelihood], np.float64)

        if self.verbose in [2, 3, 4]:
            print(-grad_loglikelihood, "\t Grad")
        if self.verbose in [4]:    
            print("Grad error is:", self.check_mlh_grad(hypers=(hypers / self.hypers_scale), verbose=0, grad1=-grad_loglikelihood))
        return -grad_loglikelihood

    
    def grad_minus_loglikelihood_findiff(self, eps=1e-6, hypers=None):
        assert self.X is not None, "fit gp before!"
        if hypers is None:
            hypers = self.hypers
        grad2 = np.zeros(hypers.shape[0])
        for i in range(hypers.shape[0]):
            tmp_hypers = hypers.copy()
            tmp_hypers[i] = hypers[i] * (1 + eps)
            f_xplus  = self.minus_loglikelihood2(tmp_hypers)
            tmp_hypers[i] = hypers[i] * (1 - eps)
            f_xminus = self.minus_loglikelihood2(tmp_hypers)
            grad2[i] = (f_xplus - f_xminus) / (2 * eps * hypers[i]) 
        return grad2


    def optimize_hypers(self, hypers_init=None):
        if hypers_init is None:
            hypers_init = self.hypers_init
        hypers_init = np.array([np.float64(x) for x in hypers_init], np.float64)
        if self.kernel_grad is None:
            hypers = opt.minimize(self.minus_loglikelihood,
                              hypers_init,           #!!!!!!
                              method='nelder-mead',
                              options={'xtol': 1e-5, 'ftol': 1e-5, 'disp': False}).x                    #!!!!!!
        else:
            hypers = opt.minimize(self.minus_loglikelihood2,
                              hypers_init,           #!!!!!!
                              method='L-BFGS-B',
                              jac=self.grad_minus_loglikelihood2,
                              options={'gtol': 1e-5, 'ftol': 1e-5, 'disp': False}).x
        return np.array([mp.mpf(h) for h in hypers])
    
    
    def optimize_hypers2(self, hypers_init=None):
        if hypers_init is None:
            hypers_init = self.hypers_init
        hypers_init = np.array([np.float64(x) for x in hypers_init], np.float64)
        hypers_nm = opt.minimize(self.minus_loglikelihood,
                          hypers_init,           #!!!!!!
                          method='nelder-mead',
                          options={'xtol': 1e-5, 'ftol': 1e-5, 'disp': False}).x                    #!!!!!!
        hypers = hypers_nm
        if self.kernel_grad is not None:
            hypers_grad = opt.minimize(self.minus_loglikelihood2,
                                  hypers_init,           #!!!!!!
                                  method='L-BFGS-B',
                                  jac=self.grad_minus_loglikelihood2,
                                  options={'gtol': 1e-5, 'ftol': 1e-5, 'disp': False}).x
            if self.minus_loglikelihood2(hypers=hypers_grad) < self.minus_loglikelihood2(hypers=hypers_nm):
                hypers = hypers_grad
        return np.array([mp.mpf(mp.fabs(h)) for h in hypers])

    def optimize_hypers_nm(self, hypers_init=None, opt_options={'xtol': 1e-5, 'ftol': 1e-5, 'disp': False}):
        if hypers_init is None:
            hypers_init = self.hypers_init
        hypers_init = np.array([np.float64(x) for x in hypers_init], np.float64)
        hypers = opt.minimize(self.minus_loglikelihood,
                          hypers_init,           #!!!!!!
                          method='nelder-mead',
                          options=opt_options).x                    #!!!!!!
        return np.array([mp.mpf(mp.fabs(h)) for h in hypers])
    
    def optimize_hypers_lbfgsb(self, hypers_init=None, opt_options={'gtol': 1e-5, 'ftol': 1e-5, 'disp': False}):
        if hypers_init is None:
            hypers_init = self.hypers_init
        hypers_init = np.array([np.float64(x) for x in hypers_init], np.float64)
        hypers = opt.minimize(self.minus_loglikelihood2,
                          hypers_init,           #!!!!!!
                          method='L-BFGS-B',
                          jac=self.grad_minus_loglikelihood2,
                          options=opt_options).x
        return np.array([mp.mpf(mp.fabs(h)) for h in hypers])

    def optimize_hypers_bfgs(self, hypers_init=None, opt_options={'gtol': 1e-5, 'norm': float('-inf'), 'disp': False}):
        if hypers_init is None:
            hypers_init = self.hypers_init
        hypers_init = np.array([np.float64(x) for x in hypers_init], np.float64)
        hypers = opt.minimize(self.minus_loglikelihood2,
                          hypers_init,           #!!!!!!
                          method='BFGS',
                          jac=self.grad_minus_loglikelihood2,
                          options=opt_options).x
        return np.array([mp.mpf(mp.fabs(h)) for h in hypers])


    def optimize_hypers_cg(self, hypers_init=None, opt_options={'gtol': 1e-5, 'norm': float('-inf'), 'disp': False}):
        if hypers_init is None:
            hypers_init = self.hypers_init
        hypers_init = np.array([np.float64(x) for x in hypers_init], np.float64)
        hypers = opt.minimize(self.minus_loglikelihood2,
                          hypers_init,           #!!!!!!
                          method='CG',
                          jac=self.grad_minus_loglikelihood2,
                          options=opt_options).x
        return np.array([mp.mpf(mp.fabs(h)) for h in hypers])
 
    
    
    def update_data(self):
        self.hypers_scaled = self.hypers * self.hypers_scale
        self.K = self.kernel_mat(self.X, self.X)
        self.cov_mat = self.K + mp.eye(self.K.rows)*self.sigma_noise**2
        if self.dy is not None:
            self.cov_mat += mp.diag(self.dy)**2
        self.cov_mat_inv = self.cov_mat**(-1)
        self.alpha = self.cov_mat_inv * mp.matrix(self.y)
     
    def fit(self, X, y, dy=None, optimize_hypers=True):
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y)
        self.dy = copy.deepcopy(dy)
        if optimize_hypers:
            try:
                self.hypers = self.optimize_hypers()
            except:
                print("Unsuccessful hypers optimisation!")
                print("hypers are", [round(x,2) for x in self.hypers])
        self.update_data()
        self.update_descendant_data()

    def fit2(self, X, y, dy=None, optimize_hypers=True):
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y)
        self.dy = copy.deepcopy(dy)
        if optimize_hypers:
            try:
                self.hypers = self.optimize_hypers2()
            except:
                print("Unsuccessful hypers optimisation!")
                print("hypers are", [round(x,2) for x in self.hypers])
        self.update_data()
        self.update_descendant_data()
        
    def predict(self, Xs):
        Ks  = self.kernel_mat(Xs, self.X)
        Kss = self.kernel_mat(Xs, Xs)
        
        fs = Ks * self.alpha
        fs_cov_mat = Kss + mp.eye(Kss.rows)*self.sigma_noise**2 - Ks*self.cov_mat_inv*Ks.T
        return fs, fs_cov_mat

    def update_descendant_data(self):
        pass
    
    def get_new_err(self):
        assert False, "get_new_error not implemented! Must be in descendant class"
        
    def added_info(self, x):
        assert self.curr_err is not None, "curr_err not implemented! Must be in descendant class"
        if not isinstance(x, list):
            x = [x_i for x_i in x]
        return mp.log(np.abs(self.get_new_err(x) / self.curr_err))     
    
    
    
    
    
    def check_kernel_grad(self, x1, x2, eps=1e-6, verbose=1):
        assert self.kernel_grad is not None, "kernel_grad is not passed!"
        grad1 = self.kernel_grad(x1, x2, self.hypers_scaled)
        grad2 = np.zeros(self.hypers_scaled.shape)
        for i in range(self.hypers_scaled.shape[0]):
            tmp_hypers = self.hypers_scaled.copy()
            tmp_hypers[i] = self.hypers_scaled[i] * (1 + eps)
            f_xplus  = self.kernel(x1, x2, tmp_hypers)
            tmp_hypers[i] = self.hypers_scaled[i] * (1 - eps)
            f_xminus = self.kernel(x1, x2, tmp_hypers)
            grad2[i] = (f_xplus - f_xminus) / (2 * eps * self.hypers_scaled[i])
        #if verbose >= 1:
#             print("kernel gradient:")
#             for i in grad1:
#                 print(i)
#             print("kernel gradient by central differences:")
#             for i in grad2:
#                 print(i)        
#            print("from func", np.array2string(a=grad1, formatter={"float":lambda x: ("%6.3e" % x) if np.abs(x) > 1e-16 else "0e0"}))
#            print("finit dif", np.array2string(a=grad2, formatter={"float":lambda x: ("%6.3e" % x) if np.abs(x) > 1e-16 else "0e0"}))
        l2 = np.zeros(np.shape(grad1))
        print_minuses_prefix = [False, False, False]
        print_minuses_postfix = [False, False, False]
        for i in range(len(grad1)):
            l2[i] += np.abs(1 - float(grad2[i] / grad1[i])) if (grad1[i]**2 > 1e-20) else 0
            l2[i] += np.abs(1 - float(grad1[i] / grad2[i])) if (grad2[i]**2 > 1e-20) else 0
            
            if (grad1[i] if (grad1[i]**2 > 1e-20) else 0) < 0:
                print_minuses_prefix[0] = True
            if (grad2[i] if (grad2[i]**2 > 1e-20) else 0) < 0:
                print_minuses_prefix[1] = True
            if (l2[i] if (l2[i]**2 > 1e-20) else 0) < 0:
                print_minuses_prefix[2] = True
            if np.abs(grad1[i] if (grad1[i]**2 > 1e-20) else 0) < 1:
                print_minuses_postfix[0] = True
            if np.abs(grad2[i] if (grad2[i]**2 > 1e-20) else 0) < 1:
                print_minuses_postfix[1] = True
            if np.abs(l2[i] if (l2[i]**2 > 1e-20) else 0) < 1:
                print_minuses_postfix[2] = True
                
        print()
        for i in range(len(grad1)):
            grad1_i = ("%6.3e" % grad1[i]) if (grad1[i]**2 > 1e-20) else "0        "
            grad2_i = ("%6.3e" % grad2[i]) if (grad2[i]**2 > 1e-20) else "0        "
            l2_i    = ("%6.3e" % l2[i]) if (l2[i]**2 > 1e-20) else "0        "
            
            if (print_minuses_prefix[0] == True) and (grad1_i[0] != "-"):
                grad1_i = " " + grad1_i
            if (print_minuses_prefix[1] == True) and (grad2_i[0] != "-"):
                grad2_i = " " + grad2_i
            if (print_minuses_prefix[2] == True) and (l2_i[0] != "-"):
                l2_i = " " + l2_i
            
            if (print_minuses_postfix[0] == True) and (grad1[-3] != "-"):
                grad1_i = grad1_i + " "
            if (print_minuses_postfix[1] == True) and (grad2[-3] != "-"):
                grad2_i = grad2_i + " "
            if (print_minuses_postfix[2] == True) and (l2[-3] != "-"):
                l2_i = l2_i + " "
                
            print(i, grad1_i, grad2_i, l2_i, sep="\t")
        return np.sum(np.abs(l2))

    def check_mlh_grad(self, eps=1e-6, hypers=None, verbose=1, grad1=None):
        assert self.kernel_grad is not None, "kernel_grad is not passed!"
        assert self.X is not None, "fit gp before!"
        if hypers is None:
            hypers = self.hypers
        if grad1 is None:
            grad1 = self.grad_minus_loglikelihood2(hypers)
        grad2 = np.zeros(hypers.shape[0])
        for i in range(hypers.shape[0]):
            tmp_hypers = hypers.copy()
            tmp_hypers[i] = hypers[i] * (1 + eps)
            f_xplus  = self.minus_loglikelihood2(tmp_hypers)
            tmp_hypers[i] = hypers[i] * (1 - eps)
            f_xminus = self.minus_loglikelihood2(tmp_hypers)
            grad2[i] = (f_xplus - f_xminus) / (2 * eps * hypers[i]) 
#        if verbose >= 1:
#             print("mlh gradient:")
#             for i in grad1:
#                 print(i)
#             print("mlh gradient by central differences:")
#             for i in grad2:
#                 print(i)
#            print("from func", np.array2string(a=grad1, formatter={"float":lambda x: ("%6.3e" % x) if np.abs(x) > 1e-16 else "0e0"}))
#            print("finit dif", np.array2string(a=grad2, formatter={"float":lambda x: ("%6.3e" % x) if np.abs(x) > 1e-16 else "0e0"}))
        l2 = np.zeros(np.shape(grad1))
        print_minuses_prefix = [False, False, False]
        print_minuses_postfix = [False, False, False]
        for i in range(len(grad1)):
            l2[i] += np.abs(1 - float(grad2[i] / grad1[i])) if (grad1[i]**2 > 1e-20) else 0
            l2[i] += np.abs(1 - float(grad1[i] / grad2[i])) if (grad2[i]**2 > 1e-20) else 0
            
            if (grad1[i] if (grad1[i]**2 > 1e-20) else 0) < 0:
                print_minuses_prefix[0] = True
            if (grad2[i] if (grad2[i]**2 > 1e-20) else 0) < 0:
                print_minuses_prefix[1] = True
            if (l2[i] if (l2[i]**2 > 1e-20) else 0) < 0:
                print_minuses_prefix[2] = True
            if np.abs(grad1[i] if (grad1[i]**2 > 1e-20) else 0) < 1:
                print_minuses_postfix[0] = True
            if np.abs(grad2[i] if (grad2[i]**2 > 1e-20) else 0) < 1:
                print_minuses_postfix[1] = True
            if np.abs(l2[i] if (l2[i]**2 > 1e-20) else 0) < 1:
                print_minuses_postfix[2] = True
        if verbose != 0:        
            print()
            for i in range(len(grad1)):
                grad1_i = ("%6.3e" % grad1[i]) if (grad1[i]**2 > 1e-20) else "0        "
                grad2_i = ("%6.3e" % grad2[i]) if (grad2[i]**2 > 1e-20) else "0        "
                l2_i    = ("%6.3e" % l2[i]) if (l2[i]**2 > 1e-20) else "0        "

                if (print_minuses_prefix[0] == True) and (grad1_i[0] != "-"):
                    grad1_i = " " + grad1_i
                if (print_minuses_prefix[1] == True) and (grad2_i[0] != "-"):
                    grad2_i = " " + grad2_i
                if (print_minuses_prefix[2] == True) and (l2_i[0] != "-"):
                    l2_i = " " + l2_i

                if (print_minuses_postfix[0] == True) and (grad1[-3] != "-"):
                    grad1_i = grad1_i + " "
                if (print_minuses_postfix[1] == True) and (grad2[-3] != "-"):
                    grad2_i = grad2_i + " "
                if (print_minuses_postfix[2] == True) and (l2[-3] != "-"):
                    l2_i = l2_i + " "

                print(i, grad1_i, grad2_i, l2_i, sep="\t")
        return np.sum(np.abs(l2))
 
