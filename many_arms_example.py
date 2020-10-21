import numpy as np
import logging
from RAGE import RAGE
from XY_ORACLE import XY_ORACLE
from XY_ADAPTIVE import XY_ADAPTIVE
from LAZY_TS import LAZY_TS
import pickle
import os
import sys
import functools
import multiprocessing as multiprocess

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# create folder for data
data_dir = os.path.join(os.getcwd(), 'direction_data_dir')
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)


# Calling algorithms
def sim_wrapper(item_list, seed_list, count):
    item_list[count].algorithm(seed_list[count])
    return item_list[count]


# Create a linear bandit problems (arms, mu)
def many_arm_problem_instance(n):
    d = 2
    x = .1 * np.random.rand(n - 2)
    arm1 = [
        [np.cos(.78 + x[i]), np.sin(.78 + x[i])] + [0 for _ in range(d - 2)]
        for i in range(n - 2)
    ]
    arm2 = [[1] + [0 for _ in range(d - 1)]]
    arm3 = [[-.707, .707] + [0 for _ in range(d - 2)]]

    X = np.vstack(arm1 + arm2 + arm3)

    theta_star = np.array([1, 0] + [0 for _ in range(d - 2)]).reshape(-1, 1)
    return X, theta_star


# parameters
count = 20
delta = 0.05
alpha = .1
eps = 0
sweep =  [1000, 2500, 5000, 7500, 10000]
factor = 10
pool_num = 2
arguments = sys.argv[1:]

# For each element in sweep is bandit problem with number of arms n
for n in sweep:
    print('Starting sweep: {}'.format(n))
    np.random.seed(43)
    X_set = []
    theta_star_set = []
    # Generate
    for i in range(count):
        X, theta_star = many_arm_problem_instance(n)
        X_set.append(X)
        theta_star_set.append(theta_star)

    # Lazy TS
    if 'lazyts' in arguments:
        print('[ALGORITHM]  * * * LTS no averaging * * *')
        np.random.seed(43)
        instance_list = [LAZY_TS(X, theta_star, delta, 2, False) for X, theta_star in zip(X_set, theta_star_set)]
        seed_list = list(np.random.randint(0, 100000, count))
        # calls the algorithm
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(pool_num)
        num_list = list(range(count))

        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            try:
                instance_list.append(instance)
                print('Finished Lazy TS instance ')
                sample_complexity = np.array([instance.tau for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity)/np.sqrt(count)
                file1 = open(os.path.join(data_dir, "tslazy_no_averaging_" + str(n) + "_data.p"), "wb")
                pickle.dump((mean, se), file1)
                file1.close()
                print('completed %d: mean %d and se %d' % (n, mean, se))
                file2 = open(os.path.join(data_dir, "tslazy_no_averaging_" + str(n) + ".p"), "wb")
                pickle.dump(instance_list, file2)
                file2.close()
            except:
                print('error')

        pool.close()
        pool.join()

        print('[ALGORITHM]  * * * LTS averaging * * *')
        np.random.seed(43)
        instance_list = [LAZY_TS(X, theta_star, delta, 2, True) for X, theta_star in zip(X_set, theta_star_set)]
        seed_list = list(np.random.randint(0, 100000, count))
        # calls the algorithm
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(pool_num)
        num_list = list(range(count))

        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            try:
                instance_list.append(instance)
                print('Finished Lazy TS instance ')
                sample_complexity = np.array([instance.tau for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity)/np.sqrt(count)
                file1 = open(os.path.join(data_dir, "tslazy_averaging_" + str(n) + "_data.p"), "wb")
                pickle.dump((mean, se), file1)
                file1.close()
                print('completed %d: mean %d and se %d' % (n, mean, se))
                file2 = open(os.path.join(data_dir, "tslazy_averaging_" + str(n) + ".p"), "wb")
                pickle.dump(instance_list, file2)
                file2.close()
            except:
                print('error')

        pool.close()
        pool.join()
        #sys.exit('done')
    # RAGE
    if 'rage' in arguments:
        print('[ALGORITHM]  * * * RAGE * * *')
        np.random.seed(43)
        instance_list = [
            RAGE(X, theta_star, factor, delta)
            for X, theta_star in zip(X_set, theta_star_set)
        ]
        seed_list = list(np.random.randint(0, 100000, count))
        # calls the algorithm
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(pool_num)
        num_list = list(range(count))

        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            try:
                instance_list.append(instance)
                print('Finished RAGE Instance')
                sample_complexity = np.array(
                    [instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity) / np.sqrt(count)
                file1 = open(
                    os.path.join(data_dir, "rage_" + str(n) + "_data.p"), "wb")
                pickle.dump((mean, se), file1)
                file1.close()
                print('completed %d: mean %d and se %d' % (n, mean, se))
                file2 = open(
                    os.path.join(data_dir, "rage_" + str(n) + ".p"), "wb")
                pickle.dump(instance_list, file2)
                file2.close()
            except:
                print('error')

        pool.close()
        pool.join()

    # XY
    if 'xy' in arguments:
        np.random.seed(43)
        instance_list = [
            XY_ADAPTIVE(X, theta_star, alpha, delta) for i in range(count)
        ]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(5)
        num_list = list(range(count))

        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            try:
                instance_list.append(instance)
                print('Finished XY Instance')
                sample_complexity = np.array(
                    [instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity) / np.sqrt(count)
                pickle.dump((mean, se),
                            open(
                                os.path.join(data_dir,
                                             "xy_" + str(d) + "_data.p"),
                                "wb"))
                print('completed %d: mean %d and se %d' % (d, mean, se))
                pickle.dump(
                    instance_list,
                    open(os.path.join(data_dir, "xy_" + str(d) + ".p"), "wb"))
            except:
                print('error')

        pool.close()
        pool.join()

    # ORACLE
    if 'oracle' in arguments:
        np.random.seed(43)
        instance_list = [
            XY_ORACLE(X, theta_star, delta)
            for X, theta_star in zip(X_set, theta_star_set)
        ]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(pool_num)
        num_list = list(range(count))

        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            try:
                instance_list.append(instance)
                print('Finished ORACLE Instance')
                sample_complexity = np.array(
                    [instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity) / np.sqrt(count)
                file1 = open(
                    os.path.join(data_dir, "oracle_" + str(n) + "_data.p"),
                    "wb")
                pickle.dump((mean, se), file1)
                file1.close()
                print('completed %d: mean %d and se %d' % (n, mean, se))
                file2 = open(
                    os.path.join(data_dir, "oracle_" + str(n) + ".p"), "wb")
                pickle.dump(instance_list, file2)
                file2.close()
            except:
                print('error')

        pool.close()
        pool.join()
