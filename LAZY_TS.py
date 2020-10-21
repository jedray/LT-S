import numpy as np
import time
import itertools
import logging
import sys
import os


class LAZY_TS(object):
    def __init__(self, X, mu, delta, laziness_factor=2, averaging=True):
        # Project in case X does not span R^d
        self.X = X
        self.mu = mu.flatten()
        #self.X, self.mu, self.d = self.__project(X,mu)
        self.K, self.d = X.shape
        self.delta = delta
        self.best_arm = np.argmax(self.X @ mu)
        self.success = False
        self.mu_hat = self.mu
        self.A = np.zeros([self.d, self.d])
        self.design = np.zeros(self.K)
        self.allocation = np.zeros(self.K)
        self.cumulative_design = np.zeros(self.K)
        self.i0 = 0
        self.A0 = np.zeros(self.d).astype(int)
        self.c0 = 0 # forced exploration constant
        self.c1 = 0
        self.c2 = 0
        self.g = self.d
        self.laziness_factor = laziness_factor;
        self.averaging = averaging;
        self.seed = 1000
        self.mu_hat = np.zeros(self.d)
        self.best_arm_hat = 0
        self.run_time = 0.
        self.support_history = []
        self.range_arms = [i for i in range(self.K-1)]
        self.Y_A = None
        self.Y_bar = None
        self.vect_diag = np.vectorize(self._diag)
        self.count_numerical_errors = 0;
        self.psi_inv = 0
        self.u = 0.1;
        self.count_fe = 0;

    def _diag(self, k):
        return np.dot(self.Y_A[k, :], self.Y_bar[k, :])

    def __project(self, X, mu):
        r = np.linalg.matrix_rank(X);
        U, D, V = np.linalg.svd(np.array(X, dtype=np.float32, order='C'));
        newX = U[:,:r]@np.diag(D[:r]);
        newmu = (V[:r,:]@mu).flatten();
        return newX, newmu, r;

    def __build_A0(self):
        r = 0
        k = 0
        while r < self.d:
            arm = np.random.randint(self.K)
            if np.linalg.matrix_rank(self.A +
                                     np.outer(self.X[arm], self.X[arm])) > r:
                self.A += np.outer(self.X[arm], self.X[arm])
                self.A0[r] = arm
                r += 1
                #logging.debug('current rank %s' % str(r))
        self.c0 = np.min(np.linalg.eigvals(self.A)) / np.sqrt(self.d)
        self.c1 = np.min(np.linalg.eigvals(self.A))

    def __Lazy_update(self, t):
        if t == int(self.g):
            self.g = np.max([np.ceil(self.laziness_factor *t), t+1])
            self.design, rho = self.__optimal_allocation(self.design)
            self.psi_inv = 2 *rho;


    def __sampling_rule(self, t):
        f = self.c0 * np.sqrt(t);
        if np.min(np.linalg.eigvals(self.A)) < f:
            arm = self.A0[self.i0]
            self.i0 = np.mod(self.i0 + 1, self.d)
            self.count_fe += 1;
        else:
            if self.averaging:
                # Averaging
                self.cumulative_design += self.design
                self.support = (self.cumulative_design > 0)
                self.map_support = np.array(range(self.K))[self.support]
                index = np.argmin((self.allocation -
                                   self.cumulative_design)[self.map_support])
            else:
                # Agressive update
                self.support = (self.design > 0)
                self.map_support = np.array(range(self.K))[self.support]
                index = np.argmin(
                    (self.allocation - t * self.design)[self.map_support])
            arm = self.map_support[index]
            self.support_history.append(np.sum(self.support))
        return arm


    def __Z(self, t):
        # We compute rho in the following manner to avoid numerical errors
        rewards = self.X @ self.mu_hat
        gaps = rewards[self.best_arm_hat] - rewards
        gaps = np.delete(gaps, self.best_arm_hat, 0)
        gaps = gaps.reshape(len(gaps), 1)
        if np.any(gaps == 0):
            return 0
        Y = self.X[self.best_arm_hat, :] - self.X
        Y = np.delete(Y, self.best_arm_hat, 0)
        Y = (1/gaps)*Y
        _I = np.ones((self.d, 1))
        design = self.allocation;
        A_inv = np.linalg.pinv(self.X.T @ np.diag(design) @ self.X)
        U, D, V = np.linalg.svd(A_inv)
        Ainvhalf = U @ np.diag(np.sqrt(D)) @ V.T
        newY = (Y @ Ainvhalf)**2
        rho = newY @ _I
        Z = (1/(2*np.max(rho)))
        return Z


    # define the stopping rule
    def __stopping_rule(self, t):
        #logging.debug('Started stopping rule')
        # theoretical threshold
        temp = np.sqrt( np.linalg.det(((1 / (self.u*self.c1)) * self.A) + np.eye(self.d)));
        beta_t = self.c2 * np.log(temp/ self.delta)
        # heuristic threshold
        #beta_t = self.c2 * (np.log(1/self.delta)+ 0.5*np.log(t) + 0.5*self.d*np.log(1/self.u + 1))
        Z_t = self.__Z(t)
        #if t % 1000 == 0:
        #    logging.critical('[round %s] stopping rule:  %s > %s and (t >= d) = %s?' % (str(t), str(Z_t), str(beta_t), str((t >= self.d))));
        return (Z_t > beta_t) and (t >= self.d)

    def algorithm(self, seed, var=True, binary=False):
        logging.critical('Started LTS')
        logging.critical('[Seed]: %s ', str(seed))
        self.var = var
        self.seed = seed
        np.random.seed(self.seed)
        t = 0
        temp = np.zeros(self.d)
        self.design = np.ones(self.K)
        self.design /= self.design.sum()
        # If K is too large, we initialize with a sparse allocation
        # self.design[1] = 0.5
        # self.design[5] = 0.5
        self.__build_A0()
        self.sigma = 1
        self.c2 = (1+self.u) * (self.sigma**2)
        logging.critical('[round %s] LTS main loop started' % str(t))
        ts = time.process_time()
        while True:
            t += 1
            pull = self.__sampling_rule(t)
            reward = self.X[pull] @ self.mu + np.random.randn(1, 1)[0][0]
            temp += reward * self.X[pull]
            self.allocation[pull] += 1
            self.A += np.outer(self.X[pull], self.X[pull])
            self.mu_hat = np.linalg.pinv(self.A) @ temp
            self.best_arm_hat = np.argmax(self.X @ self.mu_hat)
            if self.__stopping_rule(t):
                break
            self.__Lazy_update(t)
        self.run_time = time.process_time() - ts
        logging.critical('Processing time: {}'.format(self.run_time))
        logging.debug('ended mail loop')
        self.tau = t
        self.success = (self.best_arm_hat == self.best_arm)
        logging.critical('[round %s] seed %s' % (str(t), str(self.seed)))
        logging.critical('[round %s] Succeeded? %s' % (str(t), str(self.success)))
        logging.critical('[round %s] Sample complexity %s' % (str(t),str(self.tau)))
        logging.critical('[round %s] Estimated characteristic time %s' % (str(t), str(self.psi_inv)))
        logging.critical('[round %s] support %s' % (str(t), str(np.sum(self.support))))
        del self.X


    def __optimal_allocation(self, init):

        #design = init
        design = np.ones(self.K)
        design /= design.sum()
        # maximum number of iterations
        max_iter = 1000
        # construct Y
        rewards = self.X @ self.mu_hat
        gaps = np.max(rewards) - rewards
        gaps = np.delete(gaps, self.best_arm_hat, 0)
        gaps = gaps.reshape(len(gaps), 1)
        Y = self.X[self.best_arm_hat, :] - self.X
        Y = np.delete(Y, self.best_arm_hat, 0)
        Y = (1 / gaps) * Y
        _I = np.ones((self.d, 1))
        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(self.X.T @ np.diag(design) @ self.X)
            U, D, V = np.linalg.svd(A_inv)
            Ainvhalf = U @ np.diag(np.sqrt(D)) @ V.T

            newY = (Y @ Ainvhalf)**2
            rho = newY @ _I  # np.ones((newY.shape[1], 1))

            idx = np.argmax(rho)
            y = Y[idx, :, None]
            g = ((self.X @ A_inv @ y) * (self.X @ A_inv @ y)).flatten()
            g_idx = np.argmax(g)

            gamma = 2 / (count + 2)
            design_update = -gamma * design
            design_update[g_idx] += gamma

            relative = np.linalg.norm(design_update) / (np.linalg.norm(design))

            design += design_update

            if count % 100 == 0:
                logging.debug('design status %s, %s, %s, %s' %
                              (self.seed, count, relative, np.max(rho)))

            if relative < 0.01:
                logging.debug('design status %s, %s, %s, %s' %
                              (self.seed, count, relative, np.max(rho)))
                break

        idx_fix = np.where(design < 1e-5)[0]
        design[idx_fix] = 0
        design /= np.sum(design)
        return design, np.max(rho)
