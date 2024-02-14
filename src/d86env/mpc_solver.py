import time

import numpy as np
from matplotlib import pyplot as plt

# from ref_path import ref_path
from qpsolvers import solve_qp


class MpcSolver:

    def __init__(self, A: np.ndarray, B: np.ndarray, state_limit: list, normal_weight: float, end_weight: float,
                 ref_path, steps: int):
        """

        @param state_limit: 状态量的约束范围
        @param normal_weight:非终点位置的点的权重
        @param end_weight:终点位置的点的权重
        @param ref_path:参考路径
        @param steps:预测的步长
        """
        self.result = None
        self.N_ = steps
        self.ref_path_ = np.array(ref_path)
        self.B_ = B
        self.A_ = A
        self.m = B.shape[1]  # 2 输入量的数量
        self.n = A.shape[1]  # 6 状态量的数量
        self.x0 = np.zeros(shape=(self.N_ * self.n, 1))
        self.BB = np.zeros(shape=(self.N_ * self.n, self.N_ * self.m))
        self.AA = np.zeros(shape=(self.N_ * self.n, self.n))
        self.step = self.ref_path_.shape[0] / (1.0 * self.N_)
        self.x_ref = np.zeros(shape=(self.N_ * self.n, 1))
        # min J=(X-X_ref)T * Qx * (X-X_ref)
        # X:N组状态变量
        # X_ref:N组状态变量的参考点
        self.Qx = np.zeros(shape=(self.N_ * self.n, self.N_ * self.n))
        for i in range(0, self.N_ - 1, 1):
            # px 对应的权重
            self.Qx[i * self.n, i * self.n] = normal_weight
            # py 对应的权重
            self.Qx[i * self.n + 3, i * self.n + 3] = normal_weight
        self.Qx[(self.N_ - 1) * self.n, (self.N_ - 1) * self.n] = end_weight
        self.Qx[(self.N_ - 1) * self.n + 3, (self.N_ - 1) * self.n + 3] = end_weight
        # min J=0.5*uTPu + qTu
        # Gu <= h
        # Au = b
        # lb<=u<=ub
        self.P = None  # 二次项系数矩阵
        self.q = None  # 一次项矩阵
        self.G = None  # 不等式约束矩阵
        self.h = None  # 不等式约束向量
        self.lb = -state_limit[-1] * np.ones(shape=self.N_ * self.m)  # 优化变量上限
        self.ub = state_limit[-1] * np.ones(shape=self.N_ * self.m)  # 优化变量下限

    def solve_qp(self, x0):
        """

        @return:
        """

        for i in range(self.N_):
            self.x0[i * self.n:(i + 1) * self.n] = x0
            self.x_ref[i * self.n] = self.ref_path_[int(self.step * i), 0, 0] # px
            self.x_ref[i * self.n + 1] = self.ref_path_[int(self.step * i), 1, 0] # vx
            self.x_ref[i * self.n + 2] = self.ref_path_[int(self.step * i), 2, 0] # ax
            self.x_ref[i * self.n + 3] = self.ref_path_[int(self.step * i), 0, 1] # py
            self.x_ref[i * self.n + 4] = self.ref_path_[int(self.step * i), 1, 1] # vy
            self.x_ref[i * self.n + 5] = self.ref_path_[int(self.step * i), 2, 1] # ay

            if i == 0:
                self.BB[0:self.n, 0:self.m] = self.B_.copy()
                self.AA[0:self.n, 0:self.n] = self.A_.copy()
            else:
                self.AA[i * self.n:(i + 1) * self.n, 0:self.n] = self.A_ @ self.AA[(i - 1) * self.n:i * self.n,
                                                                           0:self.n]
                self.BB[i * self.n:(i + 1) * self.n, i * self.m:(i + 1) * self.m] = self.B_.copy()
                for j in range(i):
                    self.BB[i * self.n:(i + 1) * self.n, j * self.m:(j + 1) * self.m] = \
                        self.A_ @ self.BB[(i - 1) * self.n:i * self.n, j * self.m:(j + 1) * self.m]
        self.P = self.BB.T @ self.Qx @ self.BB
        self.q = self.BB.T @ self.Qx.T @ (self.x0 - self.x_ref)
        # self.result = solve_qp(self.P, self.q.T, self.G, self.h, lb=self.lb, ub=self.ub, solver="osqp",
        #                        verbose=False).reshape(self.N_ * self.m, 1)
        self.result = solve_qp(self.P, self.q.T, self.G, self.h, lb=self.lb, ub=self.ub, solver="quadprog",
                               verbose=False).reshape(self.N_ * self.m, 1)
        if self.result is None:
            return False
        return True

    def states(self):
        assert self.result is not None, "you need to solve() the MPC problem first"
        # X = np.zeros((self.nb_timesteps + 1, self.x_dim))
        p = []
        v = []
        a = []
        j = []

        X = self.BB @ self.result + self.x0
        for k in range(self.N_):
            j.append((self.result[self.m * k], self.result[self.m * k + 1]))
            p.append((X[self.n * k], X[self.n * k + 3]))
            v.append((X[self.n * k + 1], X[self.n * k + 4]))
            a.append((X[self.n * k + 2], X[self.n * k + 5]))

        return p, v, a, j


if __name__ == "__main__":
    start = np.array(
        [1.436231017112732, 0.4554591774940491, 0, 12.424905776977539, -0.06441496312618256, 0.0]).reshape(6, 1)
    dt = 0.01
    steps = 50
    A = np.array([[1., dt, dt ** 2 / 2., 0., 0., 0.],
                  [0., 1., dt, 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0.],
                  [0., 0., 0., 1., dt, dt ** 2 / 2.],
                  [0., 0., 0., 0., 1., dt],
                  [0., 0., 0., 0., 0., 1.]])
    B = np.array([[dt ** 3 / 6., 0.],
                  [dt ** 2 / 2., 0.],
                  [dt, 0.],
                  [0., dt ** 3 / 6.],
                  [0., dt ** 2 / 2.],
                  [0., dt]])
    ref_path = [[(1.4728878354147859, 12.097789760714159), (0.7514952373161028, -0.8127539427640484), (0.414049363075255, -0.7997176772247541)], [(1.5109763104510379, 12.056173131552171), (0.7719619877344635, -0.8514901108093704), (0.4042386445812106, -0.7491909748594432)], [(1.550074919896432, 12.012684154750449), (0.791882125589424, -0.8876226564566909), (0.3922093549991603, -0.6956412448960212)], [(1.59015362114509, 11.96745661905442), (0.8111483964955724, -0.9210106978648025), (0.37810846108866913, -0.6394792688824075)], [(1.6311771925803087, 11.92063084382462), (0.8296608944054762, -0.9515338922698938), (0.3620829296093018, -0.5811158283665208)], [(1.673105600991466, 11.872352652082853), (0.8473270616096807, -0.9790924359855491), (0.3442797273206234, -0.5209617048962805)], [(1.7158943689909139, 11.822772343558304), (0.8640616887367094, -1.0036070644027482), (0.32484582098219883, -0.45942768001960543)], [(1.759494942430882, 11.772043667733678), (0.8797869147530641, -1.0250190519898674), (0.3039281773535928, -0.39692453528441474)], [(1.8038550578203738, 11.72032279689132), (0.894432226963225, -1.0432902122926795), (0.2816737631943705, -0.33386305223862756)], [(1.848919109742066, 11.667767299159362), (0.9079344610096505, -1.0584028979343525), (0.25822954526409664, -0.27065401243016285)], [(1.8946285182692075, 11.614535111557828), (0.9202378008727766, -1.07036000061545), (0.23374249032233624, -0.20770819740693963)], [(1.9409220963825193, 11.56078351304479), (0.9312937788710187, -1.0791849511139322), (0.2083595651286542, -0.1454363887168772)], [(1.9877364173870913, 11.506668097562487), (0.9410612756607695, -1.0849217192851555), (0.1822277364426154, -0.08424936790789439)], [(2.0350061823292833, 11.45234174708345), (0.9495065202364004, -1.0876348140618717), (0.1554939710237849, -0.02455791652791063)], [(2.082664588221107, 11.397953610429138), (0.9566031706784336, -1.087408706204184), (0.12831169548555418, 0.03327336387874386)], [(2.130643717259124, 11.343648232173585), (0.9623332023834441, -1.0843414785490524), (0.10086017585662158, 0.08906459178209222)], [(2.1788749793371567, 11.28956506253207), (0.9666877962939593, -1.078538476259803), (0.0733251380195121, 0.14268206565574637)], [(2.2272895755535034, 11.23583811733478), (0.9696674196466344, -1.0701117295760791), (0.0458923078567508, 0.19399208397331785)], [(2.27581896252565, 11.18259564377303), (0.9712818259722492, -1.0591799538138458), (0.01874741125086269, 0.2428609452084186)], [(2.324395316704973, 11.129959786145431), (0.9715500550957104, -1.0458685493653852), (-0.007923825915627201, 0.2891549478346602)]]



    mpc = MpcSolver(A=A, B=B, state_limit=[1, 10, 1000], normal_weight=1, end_weight=10, ref_path=ref_path,
                    steps=steps)
    start_time = time.time()

    mpc.solve_qp(x0=start)
    print(time.time() - start_time)
    p, v, a, j = mpc.states()

    ## 可视化
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    axs[0, 0].plot(start[0], start[3], 'ro')
    axs[0, 0].plot([xy[0] for xy in p], [xy[1] for xy in p], 'r')
    axs[0, 0].plot(ref_path[0][0][0], ref_path[0][0][1], 'bo')
    axs[0, 0].plot([s[0][0] for s in ref_path], [s[0][1] for s in ref_path], 'b')
    axs[0, 0].set_title('pos')
    axs[1, 0].plot(start[1], start[4], 'ro')
    axs[1, 0].plot([xy[0] for xy in v], [xy[1] for xy in v], 'r')
    axs[1, 0].plot(ref_path[0][1][0], ref_path[0][1][1], 'bo')
    axs[1, 0].plot([s[1][0] for s in ref_path], [s[1][1] for s in ref_path], 'b')
    axs[1, 0].set_title('v')
    axs[1, 1].plot(start[2], start[5], 'ro')
    axs[1, 1].plot([xy[0] for xy in a], [xy[1] for xy in a], 'r')
    axs[1, 1].plot(ref_path[0][2][0], ref_path[0][2][1], 'bo')
    axs[1, 1].plot([s[2][0] for s in ref_path], [s[2][1] for s in ref_path], 'b')
    axs[1, 1].set_title('a')
    plt.show()
    pass
