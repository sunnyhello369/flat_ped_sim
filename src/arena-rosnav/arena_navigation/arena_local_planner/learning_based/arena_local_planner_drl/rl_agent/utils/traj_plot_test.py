import sys
import matplotlib.pyplot as plt

sys.path.append(r"/home/chen/desire_10086/flat_ped_sim/src/d86env")
from cubic_bspline import quniform_clamped_bspline2D



def load_reference_seeds(ctrlp_file):
    ctrl_xs = []
    ctrl_ys = []
    with open(ctrlp_file,"r") as ref:
        for line in ref.readlines():
            e = line.split("\t")
            for i in range(len(e)):
                if i < 8:
                    ctrl_xs.append(float(e[i]))
                else:
                    if i != len(e) - 1:
                        ctrl_ys.append(float(e[i]))
                    else:
                        ctrl_ys.append(float(e[i].split("\n")[0]))
            # x = float(line.split("\t")[0])
            # y = float(line.split("\t")[1])
            # yaw = float(line.split("\t")[2].split("\n")[0])
    return ctrl_xs,ctrl_ys

ctrl_xs,ctrl_ys = load_reference_seeds("/home/chen/desire_10086/flat_ped_sim/src/ctrl.txt")
plt.ion()

max_v = -3.
min_v = 3.
max_a = -2.5
min_a= 2.5
max_j = -4.5
min_j = 4.5
for j in range(0,len(ctrl_ys),8):
    plt.clf()

    p_figure = plt.subplot(221)
    v_figure = plt.subplot(222)
    a_figure = plt.subplot(223)
    j_figure = plt.subplot(224)

    p_figure.set_title('p')
    p_figure.set_xlabel('x')
    p_figure.set_ylabel('y')
    v_figure.set_title('v')
    v_figure.set_xlabel('vx')
    v_figure.set_ylabel('vy')
    a_figure.set_title('a')
    a_figure.set_xlabel('ax')
    a_figure.set_ylabel('ay')
    j_figure.set_title('j')
    j_figure.set_xlabel('jx')
    j_figure.set_ylabel('jy')

    x = []
    y = []
    vx = []
    vy = []
    ax = []
    ay = []
    jx = []
    jy = []

    ctrlx = []
    ctrly = []
    for i in range(8):
        ctrlx.append(ctrl_xs[j+i])
        ctrly.append(ctrl_ys[j+i])
    traj_spline_world = quniform_clamped_bspline2D(ctrlx,ctrly,5,3.)

    for i in range(51):
        wx,wy = traj_spline_world.calc_position_u(i*3./50.)
        x.append(wx)
        y.append(wy)
        wvx,wvy = traj_spline_world.calcd(i*3./50.)
        vx.append(wvx)
        vy.append(wvy)
        wax,way = traj_spline_world.calcdd(i*3./50.)
        ax.append(wax)
        ay.append(way)
        wjx,wjy = traj_spline_world.calcddd(i*3./50.)
        jx.append(wjx)
        jy.append(wjy)
        if wvx > max_v:
            max_v =  wvx
        elif wvx < min_v:
            min_v = wvx
        if wvy > max_v:
            max_v =  wvy
        elif wvy < min_v:
            min_v = wvy

        if wax > max_a:
            max_a =  wax
        elif wax < min_a:
            min_v = wax
        if way > max_a:
            max_a =  way
        elif way < min_a:
            min_a = way

        if wjx > max_j:
            max_j =  wjx
        elif wjx < min_j:
            min_j = wjx
        if wjy > max_j:
            max_j =  wjy
        elif wjy < min_j:
            min_j = wjy

    p_figure.plot(x,y,"r")
    v_figure.plot(vx,vy,"g")
    a_figure.plot(ax,ay,"b")
    j_figure.plot(jx,jy,"k")

    if max_v > 2.5 or min_v < -2.5 or max_a > 2. or min_a < 2. or max_j > 4. or min_j < 4.:
        plt.pause(0.1)
    else:
        plt.pause(0.1)

print("max v:")
print(max_v)
print("min v:")
print(min_v)
print("max a:")
print(max_a)
print("min a:")
print(min_a)
print("max j:")
print(max_j)
print("min j:")
print(min_j)