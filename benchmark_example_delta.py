import numpy as np
import logging
from RAGE import RAGE
from XY_ADAPTIVE import XY_ADAPTIVE
from ALBA_YELIM import ALBA_YELIM
from XY_ORACLE import XY_ORACLE
from LAZY_TS import LAZY_TS
import pickle
import os
import sys
import functools
import multiprocessing as multiprocess

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

data_dir = os.path.join(os.getcwd(), 'sin_cos_data_dir')

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

def sim_wrapper(item_list, seed_list, count):
    item_list[count].algorithm(seed_list[count])
    return item_list[count]

def sin_cos_problem_instance(d, rad):
    theta_star = np.zeros((d, 1))
    theta_star[0, 0] = 2.0
    X = np.eye(d)
    tmp = np.zeros(d)
    tmp[0] = np.cos(rad)
    tmp[1] = np.sin(rad)
    X = np.r_[X, np.expand_dims(tmp, 0)]
    return X, theta_star

count = 100
rad = .1
alpha = .1
eps = 0
sweep = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
factors = [2]
factor = 10
pool_num = 1
arguments = sys.argv[1:]
d = 6;

for delta in sweep:

    X, theta_star = sin_cos_problem_instance(d, rad)

    # Lazy TS
    if 'lazyts' in arguments:
        for f in factors:
            print('[ALGORITHM]  * * * LAZY_TS * * *')
            np.random.seed(43)
            instance_list = [LAZY_TS(X, theta_star, delta, f) for i in range(count)]

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
                    sample_complexity = np.array(
                        [instance.tau for instance in instance_list])
                    mean = np.mean(sample_complexity)
                    se = np.std(sample_complexity) / np.sqrt(count)
                    file1 = open(
                        os.path.join(data_dir, "tslazy_" + str(delta) + "_" + str(f) + "_data.p"),
                        "wb")
                    pickle.dump((mean, se), file1)
                    file1.close()
                    print('completed %d: mean %d and se %d' % (d, mean, se))
                    file2 = open(
                        os.path.join(data_dir, "tslazy_" + str(delta) + "_" + str(f) + ".p"), "wb")
                    pickle.dump(instance_list, file2)
                    file2.close()
                except:
                    print('error')
            pool.close()
            pool.join()



    # RAGE
    if 'rage' in arguments:
        np.random.seed(43)
        instance_list = [
            RAGE(X, theta_star, factor, delta) for i in range(count)
        ]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(pool_num)
        num_list = list(range(count))
        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            try:
                instance_list.append(instance)

                print('Finished Rage Instance')
                sample_complexity = np.array(
                    [instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity) / np.sqrt(count)
                f = open(
                    os.path.join(data_dir, "rage_" + str(delta) + "_data.p"), "wb")
                pickle.dump((mean, se), f)
                f.close()
                print('completed %d: mean %d and se %d' % (d, mean, se))

                f = open(os.path.join(data_dir, "rage_" + str(delta) + ".p"), "wb")
                pickle.dump(instance_list, f)
                f.close()
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
        pool = multiprocess.Pool(pool_num)
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

                f = open(
                    os.path.join(data_dir, "xy_" + str(delta) + "_data.p"), "wb")
                pickle.dump((mean, se), f)
                f.close()
                print('completed %d: mean %d and se %d' % (d, mean, se))

                f = open(os.path.join(data_dir, "xy_" + str(delta) + ".p"), "wb")
                pickle.dump(instance_list, f)
                f.close()
            except:
                print('error')

        pool.close()
        pool.join()
    #
    # LINGAP
    if 'lingap' in arguments:
        np.random.seed(43)
        instance_list = [
            LIN_GAP_ELIM(X, theta_star, eps, delta) for i in range(count)
        ]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(pool_num)
        num_list = list(range(count))
        instance_list = pool.map(parallel_sim, num_list)

        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            try:
                instance_list.append(instance)

                print('Finished LINGAP Instance')
                sample_complexity = np.array(
                    [instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity) / np.sqrt(count)

                f = open(
                    os.path.join(data_dir, "lingap_" + str(delta) + "_data.p"),
                    "wb")
                pickle.dump((mean, se), f)
                f.close()
                print('completed %d: mean %d and se %d' % (d, mean, se))

                f = open(
                    os.path.join(data_dir, "lingap_" + str(delta) + ".p"), "wb")
                pickle.dump(instance_list, f)
                f.close()

            except:
                print('error')

        pool.close()
        pool.join()
    #
    # ALBA
    if 'alba' in arguments:
        np.random.seed(43)
        instance_list = [
            ALBA_YELIM(X, theta_star, delta) for i in range(count)
        ]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(pool_num)
        num_list = list(range(count))

        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            try:
                instance_list.append(instance)

                print('Finished ALBA Instance')
                sample_complexity = np.array(
                    [instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity) / np.sqrt(count)

                f = open(
                    os.path.join(data_dir, "alba_" + str(delta) + "_data.p"), "wb")
                pickle.dump((mean, se), f)
                f.close()
                print('completed %d: mean %d and se %d' % (d, mean, se))

                f = open(os.path.join(data_dir, "alba_" + str(delta) + ".p"), "wb")
                pickle.dump(instance_list, f)
                f.close()
            except:
                print('error')

        pool.close()
        pool.join()
    #
    # # STATIC
    # if 'static' in arguments:
    #     np.random.seed(43)
    #     instance_list = [XY_STATIC(X, theta_star, delta) for i in range(count)]
    #     seed_list = list(np.random.randint(0, 100000, count))
    #     parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
    #     pool = multiprocess.Pool(pool_num)
    #     num_list = list(range(count))
    #     instance_list = []
    #     for instance in pool.imap_unordered(parallel_sim, num_list):
    #         try:
    #             instance_list.append(instance)
    #
    #             print('Finished Static Instance')
    #             sample_complexity = np.array(
    #                 [instance.N for instance in instance_list])
    #             mean = np.mean(sample_complexity)
    #             se = np.std(sample_complexity) / np.sqrt(count)
    #
    #             f = open(
    #                 os.path.join(data_dir, "static_" + str(d) + "_data.p"),
    #                 "wb")
    #             pickle.dump((mean, se), f)
    #             f.close()
    #             print('completed %d: mean %d and se %d' % (d, mean, se))
    #
    #             f = open(
    #                 os.path.join(data_dir, "static_" + str(d) + ".p"), "wb")
    #             pickle.dump(instance_list, f)
    #             f.close()
    #         except:
    #             print('error')
    #
    #     pool.close()
    #     pool.join()
    #
    # ORACLE
    if 'oracle' in arguments:
        np.random.seed(43)
        instance_list = [XY_ORACLE(X, theta_star, delta) for i in range(count)]
        seed_list = list(np.random.randint(0, 100000, count))
        parallel_sim = functools.partial(sim_wrapper, instance_list, seed_list)
        pool = multiprocess.Pool(pool_num)
        num_list = list(range(count))
        instance_list = []
        for instance in pool.imap_unordered(parallel_sim, num_list):
            try:
                instance_list.append(instance)

                print('Finished Oracle Instance')
                sample_complexity = np.array(
                    [instance.N for instance in instance_list])
                mean = np.mean(sample_complexity)
                se = np.std(sample_complexity) / np.sqrt(count)

                f = open(
                    os.path.join(data_dir, "oracle_" + str(delta) + "_data.p"),
                    "wb")
                pickle.dump((mean, se), f)
                f.close()
                print('completed %d: mean %d and se %d' % (d, mean, se))

                f = open(
                    os.path.join(data_dir, "oracle_" + str(delta) + ".p"), "wb")
                pickle.dump(instance_list, f)
                f.close()
            except:
                print('error')

        pool.close()
        pool.join()
