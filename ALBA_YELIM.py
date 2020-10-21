"""
This code is from the repo https://github.com/fiezt/Transductive-Linear-Bandit-Code
"""

import numpy as np
import itertools
import logging
import time


class ALBA_YELIM(object):
    def __init__(self, X, theta_star, delta):
        self.X = X
        self.K = len(X)
        self.d = X.shape[1]
        self.theta_star = theta_star
        self.opt_arm = np.argmax(X@theta_star)
        self.delta = delta
        self.run_time = 0.



    def algorithm(self, seed, binary=False):

        self.seed = seed
        np.random.seed(self.seed)

        self.active_arms = list(range(self.K))
        self.arm_counts = np.zeros(self.K)
        self.N = 0
        r = 0
        self.norm_bound = np.linalg.norm(self.theta_star)
        A = np.zeros((self.d, self.d))
        b = np.zeros((self.d, 1))

        while len(self.active_arms) > 1:

            delta_r = 6*self.delta/(np.pi**2*(r+1)**2)
            p = np.floor(self.d/(2**r))
            s = 1
            num_round_active = len(self.active_arms)
            ts = time.process_time()

            while len(self.active_arms) > p:

                delta_s = 6*delta_r/(np.pi**2*s**2)
                self.e_s = 1/(2**s)

                c_0 = max(4*self.norm_bound**2, 3)
                l_s = 4*c_0*(2+(6 + 4/1.1**s)*self.d)*(1.1**s/4)**2
                num_samples = np.ceil(l_s*np.log(5*num_round_active**2/(2*delta_s))).astype(int)

                design = self.optimal_allocation()
                allocation = np.random.choice(self.K, num_samples, True, p=design).tolist()
                allocation = np.array([allocation.count(i) for i in range(self.K)])

                self.M = self.X.T@np.diag(design)@self.X
                self.M_inv = np.linalg.pinv(self.M)

                pulls = np.vstack([np.tile(self.X[i], (num, 1)) for i, num in enumerate(allocation) if num > 0])

                if not binary:
                    rewards = pulls@self.theta_star + np.random.randn(num_samples, 1)
                else:
                    rewards = np.random.binomial(1, pulls@self.theta_star, (num_samples, 1))

                A = num_samples*self.M
                b = pulls.T@rewards

                self.theta_hat = np.linalg.pinv(A)@b
                self.drop_arms(l_s, design)

                s += 1
                self.arm_counts += allocation
                self.N += num_samples

                logging.info('\n\n')
                logging.info('arm counts %s' % str(self.arm_counts))
                logging.info('round sample count %s' % str(num_samples))
                logging.info('total sample count %s' % str(self.N))
                logging.info('active arms %s' % str(self.active_arms))
                logging.info('\n\n')

            r += 1

            logging.info('\n\n')
            logging.info('finished elim phase %s' % str(r-1))
            logging.info('arm counts %s' % str(self.arm_counts))
            logging.info('round sample count %s' % str(num_samples))
            logging.info('total sample count %s' % str(self.N))
            logging.info('active arms %s' % str(self.active_arms))
            logging.info('\n\n')
        self.run_time = time.process_time() - ts

        del self.M_inv
        del self.M
        self.success = (self.opt_arm in self.active_arms)
        logging.critical('Succeeded? %s' % str(self.success))
        logging.critical('Sample complexity %s' % str(self.N))


    def optimal_allocation(self):

        span_arms = self.active_arms.copy()
        rank = np.linalg.matrix_rank(self.X[self.active_arms])

        for arm_idx in range(self.K):
            if arm_idx in span_arms:
                continue
            else:
                if np.linalg.matrix_rank(self.X[self.active_arms + [arm_idx]]) == rank:
                    span_arms.append(arm_idx)

        span_arms = sorted(span_arms)
        design = np.ones(len(span_arms))

        design /= design.sum()
        Xhat = self.X[span_arms]

        max_iter = 5000

        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(Xhat.T@np.diag(design)@Xhat)

            U,D,V = np.linalg.svd(A_inv)
            Ainvhalf = U@np.diag(np.sqrt(D))@V.T
            newX = (Xhat@Ainvhalf)**2
            rho = newX@np.ones((newX.shape[1], 1))

            idx = np.argmax(rho)
            x = Xhat[idx, :, None]

            g = ((Xhat@A_inv@x)*(Xhat@A_inv@x)).flatten().tolist()

            g_idx = np.argmax(g)
            gamma = 2/(count+2)
            design_update = -gamma*design
            design_update[g_idx] += gamma

            relative = np.linalg.norm(design_update)/(np.linalg.norm(design))

            design += design_update

            if np.abs(np.max(rho) - min(self.d, len(self.active_arms))) < 0.01:
                break

        design_ = np.zeros(self.K)
        design_[span_arms] = design

        idx_fix = np.where(design_ < 1e-5)[0]
        drop_total = design_[idx_fix].sum()
        design_[idx_fix] = 0
        design_[np.argmax(design_)] += drop_total

        return design_


    def drop_arms(self, l, design):

        active_arms = self.active_arms.copy()

        max_arm_idx = np.argmax(self.X[active_arms]@self.theta_hat)
        max_arm_idx = active_arms[max_arm_idx]
        max_arm = self.X[max_arm_idx, :, None]

        for arm_idx in active_arms:

            if arm_idx == max_arm_idx:
                continue

            arm = self.X[arm_idx, :, None]
            y = max_arm - arm

            if self.conf(y, l, design) < y.T@self.theta_hat:
                self.active_arms.remove(arm_idx)


    def conf(self, y, l, design):

        L = self.norm_bound
        y_var = np.sqrt(y.T@self.M_inv@y)
        y_norm = np.linalg.norm(y)

        err = (np.sqrt(2)*L*y_var)/np.sqrt(l) \
              + (2*L*(y_norm + y_var*np.sqrt(self.d)))/(3*l) \
              + np.sqrt(2)*np.sqrt((y_var**2/l)*np.sqrt(3*self.d/l) + (y_var**2/l))

        expectation = (design.reshape(-1, 1)*((y.T@self.M_inv@self.X.T).T*(np.abs(self.X@self.theta_hat) + self.e_s/2))**2).sum()

        ERR = np.sqrt(2*expectation/l) \
              + (2*(np.abs(y.T@self.theta_hat) + self.e_s/2) + L*y_var*np.sqrt(self.d))/(3*l) \
              + np.sqrt(2*((y_var**2/l)*np.sqrt(3*self.d/l) + (y_var**2/l)))

        return min(float(err), float(ERR))
