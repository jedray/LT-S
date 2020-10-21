"""
This code is from the repo https://github.com/fiezt/Transductive-Linear-Bandit-Code
"""

import numpy as np
import itertools
import logging
import time

import warnings
warnings.filterwarnings('error')

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

def randargmin(b,**kw):
    idxs = np.where(np.abs(b-b.min())<1e-20)[0]
    return np.random.choice(idxs)


class XY_ORACLE(object):

    def __init__(self, X, theta_star, delta, Z=None):

        self.X = X
        if Z is None:
            self.Z = X
        else:
            self.Z = Z
        self.K = len(X)
        self.K_Z = len(self.Z)
        self.d = X.shape[1]
        self.theta_star = theta_star
        self.opt_arm = np.argmax(self.Z@theta_star)
        rewards = self.Z@self.theta_star
        self.gaps = np.max(rewards) - rewards
        self.gaps = np.delete(self.gaps, self.opt_arm, 0)
        self.delta = delta
        self.run_time = 0.


    def algorithm(self, seed, binary=False):


        #debug_seed = 14148
        self.seed = seed
        np.random.seed(self.seed)

        self.arm_counts = np.zeros(self.K)
        self.build_Y()
        self.N = 0

        self.A = np.zeros((self.d, self.d))
        self.b = np.zeros((self.d, 1))

        stop = False
        self.u = 1.1
        self.phase_index = 1

        design, rho = self.optimal_allocation()
        logging.critical('Optimal %s' % str(rho))
        logging.critical('design: %s' % str(design))
        ts = time.process_time()
        while True:

            self.delta_t = self.delta/(2*self.phase_index**2*self.K_Z)
            num_samples = int(np.ceil(self.u**self.phase_index))
            logging.info('num samples %s' % str(num_samples))

            allocation = np.random.choice(self.K, num_samples, p=design).tolist()
            allocation = np.array([allocation.count(i) for i in range(self.K)])
            logging.info('allocation %s' % str(allocation))

            pulls = np.vstack([np.tile(self.X[i], (num, 1)) for i, num in enumerate(allocation) if num > 0])

            if not binary:
                rewards = pulls@self.theta_star + np.random.randn(num_samples, 1)
            else:
                rewards = np.random.binomial(1, pulls@self.theta_star, (num_samples, 1))

            self.A += pulls.T@pulls
            self.A_inv = np.linalg.pinv(self.A)
            self.b += pulls.T@rewards
            self.theta_hat = self.A_inv@self.b

            best_idx = self.check_stop()

            if best_idx is None:
                pass
            else:
                stop = True

            self.phase_index += 1
            self.arm_counts += allocation
            self.N += num_samples

            if self.N % 100000 == 0:
                logging.info('\n\n')
                logging.debug('arm counts %s' % str(self.arm_counts))
                logging.info('total sample count %s' % str(self.N))
                logging.info('\n\n')

            if stop:
                break

            self.phase_index += 1
        self.run_time = time.process_time() - ts
        del self.b
        del self.A
        del self.A_inv
        del self.Yhat
        self.success = (self.opt_arm == best_idx)
        logging.critical('Succeeded? %s' % str(self.success))
        logging.critical('Sample complexity %s' % str(self.N))


    def build_Y(self):

        self.Yhat = self.Z[self.opt_arm, :] - self.Z
        self.Yhat = np.delete(self.Yhat, self.opt_arm, 0)
        self.Yhat = self.Yhat/self.gaps

    def optimal_allocation(self):

        design = np.ones(self.K)
        design /= design.sum()

        max_iter = 5000

        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(self.X.T@np.diag(design)@self.X)

            U,D,V = np.linalg.svd(A_inv)
            Ainvhalf = U@np.diag(np.sqrt(D))@V.T

            newY = (self.Yhat@Ainvhalf)**2
            rho = newY@np.ones((newY.shape[1], 1))

            idx = np.argmax(rho)
            y = self.Yhat[idx, :, None]
            g = ((self.X@A_inv@y)*(self.X@A_inv@y)).flatten()
            g_idx = np.argmax(g)

            gamma = 2/(count+2)
            design_update = -gamma*design
            design_update[g_idx] += gamma

            relative = np.linalg.norm(design_update)/(np.linalg.norm(design))

            design += design_update

            if relative < 0.01:
                break

        idx_fix = np.where(design < 1e-5)[0]
        drop_total = design[idx_fix].sum()
        design[idx_fix] = 0
        design[np.argmax(design)] += drop_total

        return design, np.max(rho)


    def check_stop(self):

        stop = True

        arm = self.Z[self.opt_arm, :, None]

        for arm_idx_prime in range(self.K_Z):

            if self.opt_arm == arm_idx_prime:
                continue

            arm_prime = self.Z[arm_idx_prime, :, None]
            y = arm - arm_prime

            try:
                if np.sqrt(2*y.T@self.A_inv@y*np.log(1/self.delta_t)) >= y.T@self.theta_hat:
                    stop = False
                    break
            except:
                stop = False
                break

        if stop:
            return self.opt_arm
        else:
            None
