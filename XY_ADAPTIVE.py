"""
This code is from the repo https://github.com/fiezt/Transductive-Linear-Bandit-Code
"""

import numpy as np
import itertools
import logging
import time

def randargmin(b,**kw):
    idxs = np.where(np.abs(b-b.min())<1e-20)[0]
    return np.random.choice(idxs)


class XY_ADAPTIVE(object):
    def __init__(self, X, theta_star, alpha, delta):
        self.X = X
        self.K = len(X)
        self.d = X.shape[1]
        self.theta_star = theta_star
        self.opt_arm = np.argmax(X@theta_star)
        self.alpha = alpha
        self.delta = delta
        self.run_time = 0.



    def algorithm(self, seed):

        np.random.seed(seed)

        self.active_arms = list(range(self.K))
        self.arm_counts = np.zeros(self.K)
        self.build_Y()
        n_past = self.d*(self.d + 1) + 1
        rho = 1
        rho_past = 1
        self.N = 0
        self.phase_index = 1
        ts = time.process_time()
        while len(self.active_arms) > 1:

            n = self.d
            self.N += self.d
            self.A = np.eye(self.d)
            self.A_inv = np.linalg.inv(self.A)
            self.b = np.zeros((self.d, 1))

            while rho/n >= self.alpha*rho_past/n_past:

                arm_idx, rho = self.select_greedy_arm()
                arm = self.X[arm_idx, :, None]

                r = self.pull(arm)
                self.b += arm*r
                self.A += arm@arm.T
                self.A_inv -= (self.A_inv@arm@arm.T@self.A_inv)/(1+arm.T@self.A_inv@arm)

                n += 1
                self.arm_counts[arm_idx] += 1
                self.N += 1

            n_past = n
            rho_past = rho

            self.theta_hat = self.A_inv@self.b
            self.drop_arms()
            self.build_Y()
            self.phase_index += 1

            logging.info('\n\n')
            logging.info('finished phase %s' % str(self.phase_index-1))
            logging.info('arm counts %s' % str(self.arm_counts))
            logging.info('round sample count %s' % str(n))
            logging.info('total sample count %s' % str(self.N))
            logging.info('active arms %s' % str(self.active_arms))
            logging.info('rho %s' % str(rho))
            logging.info('\n\n')

        self.run_time = time.process_time() - ts
        del self.b
        del self.A
        del self.A_inv
        self.success = (self.opt_arm in self.active_arms)
        logging.critical('Succeeded? %s' % str(self.success))
        logging.critical('Sample complexity %s' % str(self.N))


    def build_Y(self):

        Y = []
        for x, x_prime in itertools.permutations(self.X[self.active_arms], 2):
            Y.append(x - x_prime)

        self.Yhat = np.array(Y)


    def select_greedy_arm(self):

        score = np.diag(self.Yhat@self.A_inv@self.Yhat.T).reshape(1, -1) \
                - ((self.X@self.A_inv@self.Yhat.T)**2/(1+ np.diag(self.X@self.A_inv@self.X.T).reshape(-1, 1)))

        arm_idx = randargmin(score[np.arange(self.K), np.argmax(score, axis=1)])

        rho = np.max(score[arm_idx,:])

        return arm_idx, rho


    def pull(self, arm):

        r = arm.T@self.theta_star + np.random.randn()

        return r

    def drop_arms(self):

        active_arms = self.active_arms.copy()

        for arm_idx in active_arms:

            arm = self.X[arm_idx, :, None]

            for arm_idx_prime in active_arms:

                if arm_idx == arm_idx_prime:
                    continue

                arm_prime = self.X[arm_idx_prime, :, None]
                y = arm_prime - arm

                if np.sqrt(2*y.T@self.A_inv@y*np.log(2*np.pi**2*self.K**2*self.phase_index**2/(6*self.delta))) <= y.T@self.theta_hat:
                    self.active_arms.remove(arm_idx)
                    break
