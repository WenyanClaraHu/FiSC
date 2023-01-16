from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import autograd.numpy as anp
from pymoo.core.problem import Problem
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymoo.util.remote import Remote
import numpy as np
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.util.plotting import plot
import calculate_coverage as cal_cov
import pandas as pd
import sys

class MyProblem(Problem):
    def __init__(self, n_var, k=None):
        super().__init__(n_var=n_var, n_obj=1, n_constr=1, xl=1, xu=12024, type_var=anp.int)

    def _evaluate(self, X_M, out, *args, **kwargs):

        sess, ops = cal_cov.load_model()
        df_bus = pd.read_csv('./data/busStation_80oriFishID.csv')
        df_popular = pd.read_csv('./data/popularity_class_new.csv')
        df_RtoF = pd.read_csv('./data/Shanghai_80_80_fishnet_ori.csv')
        df_f_r = pd.read_csv('./data/fishnet_to_region_80.csv')
        df_vehInfo = pd.read_csv('./data/vehInfo.csv')

        f1 = self.cal_coverage(sess, ops, X_M, 6, df_popular, df_bus, df_vehInfo, df_f_r, df_RtoF)
        g1 = self.cal_g(X_M)
        # f1 = 1/anp.sum(X_M, axis=1)
        # f2 = 1/anp.sum(anp.square(X_M - anp.mean(X_M)), axis=1)
        out["F"] = np.column_stack([f1])
        out["G"] = np.column_stack([g1])
    #
    def cal_g(self, X_M):
        lst_g = []
        for i in range(len(X_M)):
            n = len(list(set(list(X_M[i]))))
            if n > 95 & n <= 100:
                lst_g.append(-1)
            else:
                lst_g.append(1)
        return np.array(lst_g)

    def cal_coverage(self, sess, ops, X_M, tg, df_popular, df_bus, df_vehInfo, df_f_r, df_RtoF):
        lst_coverage = []
        df = pd.read_csv('./data/generalHospital.csv')
        arr_w = np.array(list(df['weight']))
        for i in range(len(X_M)):
            coverage = cal_cov.estimate(sess, ops, list(X_M[i]), tg, df_popular, df_bus, df_vehInfo, df_f_r, df_RtoF)
            coverage = coverage*arr_w
            coverage = 1/coverage.sum()
            lst_coverage.append(coverage)
        return np.array(lst_coverage)

def rank(group):
    for i in range(1, len(group)):
        for j in range(0, len(group) - i):
            if group[j].data['crowding'] > group[j + 1].data['crowding']:
                group[j], group[j + 1] = group[j + 1], group[j]
    return group

log_print = open('Defalust.log', 'w')
sys.stdout = log_print
sys.stderr = log_print

if __name__ == '__main__':
    nvar = [100, 200, 300, 400, 600, 700, 800, 900, 1000]
    for i in range(len(nvar)):
        algorithm = GA(pop_size=180, sampling=get_sampling('int'), crossover=get_crossover('int_c', prob=0.9, eta=15),
                       mutation=get_mutation('int_pm', eta=20), eliminate_duplicates=True)  # , n_offsprings=10
        res = minimize(MyProblem(n_var=nvar[i]), algorithm, ('n_gen', 1000), seed=1, verbose=True)
        print('-----------result----------------')
        print(nvar[i], list(res.X))
