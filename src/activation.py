import matplotlib.pyplot as plt
from math import *

#恒等函数
def identity(x):
    return x

def d_identity(x):
    return 1

#单位阶跃函数
def step(x):
    if x < 0:
        return 0
    else:
        return 1

def d_step(x):
    if x != 0:
        return 0
    else:
        return None

#逻辑函数
def logic(x):
    return 1/(1+exp(-x))

def d_logic(x):
    return logic(x) * (1-logic(x))

#双曲线正切函数
def tanh(x):
    return (exp(x)-exp(-x))/(exp(x)+exp(-x))

def d_tanh(x):
    return 1-tanh(x) * tanh(x)

#反正切函数
def anti_tan(x):
    return atan(x)

def d_anti_tan(x):
    return 1/(1+x*x)

#softsign函数
def softsign(x):
    return x/(1+abs(x))

def d_softsign(x):
    return 1/((1+abs(x)) * (1+abs(x)))

#参数换relu
def prelu(x, a):
    if x < 0:
        return a * x
    else:
        return x

def d_prelu(x, a):
    if x < 0:
        return a
    else:
        return 1

#线性整流函数
def relu(x):
    return prelu(x, 0)

def d_relu(x):
    return d_prelu(x, 0)

#leaky relu
def leaky_relu(x):
    return prelu(x, 0.01)

def d_leaky_relu(x):
    return d_prelu(x, 0.01)

#rrele
def rrelu(x, a):
    return prelu(x, a)

def d_rrelu(x,a):
    return d_prelu(x, a)

#elu
def elu(x, a):
    if x < 0:
        return a * (exp(x) - 1)
    else:
        return x

def d_elu(x, a):
    if x < 0:
        return elu(x,a) + a
    else:
        return 1

#selu
def selu(x, a, l):
    if x < 0:
        return l * a * (exp(x) - 1)
    else:
        x

def d_selu(x, a, l):
    if x < 0:
        return l * a * exp(x)
    else:
        return 1

#SRelu
def srelu(x,tl,al,tr,ar):
    if x <= tl:
        return tl + al * (x - tl)
    elif x < tr and x > tl:
        return x
    else:
        return tr + ar * (x - tr)


def d_srelu(x, tl, al, tr, ar):
    if x <= tl:
        return al
    elif x < tr and x > tl:
        return 1
    else:
        return ar

#isrlu
def isrlu(x, a):
    if x < 0:
        return x/(sqrt(1 + a * x * x))
    else:
        return x

def d_isrlu(x, a):
    if x < 0:
        return (1/(sqrt(1 + a * x * x))) * (1/(sqrt(1 + a * x * x))) * (1/(sqrt(1 + a * x * x)))
    else:
        return 1

#apl

#softplus
def softplus(x):
    return log(1 + exp(x))

def d_softplus(x):
    return x/(2 * sqrt(x * x + 1)) + 1

#弯曲恒等函数
def twist(x):
    return (sqrt(x*x + 1) - 1)/2 + x

def d_twist(x):
    x/(2*sqrt(x*x+1))+1

#silu
def silu(x):
    return x * logic(x)

def d_silu(x):
    return silu(x) + logic(x) * (1-silu(x))

#SoftExponential
def softexponential(x,a):
    if a == 0:
        return x
    elif a < 0:
        return -1 * log(1 - a * (x + a))/a
    else:
        return (exp(a*x) - 1)/a

def d_softexponential(x,a):
    if a < 0:
        return 1/(1-a*(a+x))
    else:
        exp(a*x)


#sin
def usin(x):
    return sin(x)

def d_usin(x):
    return cos(x)

#sinc
def sinc(x):
    if x == 0:
        return 1
    else:
        return sin(x)/x

def d_sinc(x):
    if x == 0:
        return 0
    else:
        return cos(x)/x-sin(x)/x
#gs
def gs(x):
    return exp(-1 * x * x)

def d_gs(x):
    return -2 * x * exp(-1 * x * x)

def test_no_parameter():
    des = 'gs'
    func = gs
    d_func = d_gs

    funcs = [(identity, d_identity, 'identity'),
            (step,d_step, 'step'),
            (logic,d_logic,'logic'),
            (tanh,d_tanh,'tanh'),
            (anti_tan,d_anti_tan,'anti_tan'),
            (softsign,d_softsign,'softsign'),
            (relu,d_relu,'relu'),
            (leaky_relu,d_leaky_relu,'leaky_relu'),
            (softplus,d_softplus,'softplus'),
            (twist,d_twist,'twist'),
            (silu,d_silu,'silu'),
            (usin,d_usin,'usin'),
            (sinc,d_sinc,'sinc'),
            (gs,d_gs,'gs')]
    x = [i/10 for i in range(-100,100)]
    for f in funcs:
        func,d_func,des=f
        print(des)
        y = []
        dy = []
        for e in x:
            y.append(func(e))
            dy.append(d_func(e))

        plt.figure()
        ax = plt.gca()    # 得到图像的Axes对象
        ax.spines['right'].set_color('none')   # 将图像右边的轴设为透明
        ax.spines['top'].set_color('none')     # 将图像上面的轴设为透明
        ax.xaxis.set_ticks_position('bottom')    # 将x轴刻度设在下面的坐标轴上
        ax.yaxis.set_ticks_position('left')         # 将y轴刻度设在左边的坐标轴上
        ax.spines['bottom'].set_position(('data', 0))   # 将两个坐标轴的位置设在数据点原点
        ax.spines['left'].set_position(('data', 0))

        plt.plot(x, y)
        plt.savefig('imgs/' + des + '.png')

        plt.figure()
        ax = plt.gca()    # 得到图像的Axes对象
        ax.spines['right'].set_color('none')   # 将图像右边的轴设为透明
        ax.spines['top'].set_color('none')     # 将图像上面的轴设为透明
        ax.xaxis.set_ticks_position('bottom')    # 将x轴刻度设在下面的坐标轴上
        ax.yaxis.set_ticks_position('left')         # 将y轴刻度设在左边的坐标轴上
        ax.spines['bottom'].set_position(('data', 0))   # 将两个坐标轴的位置设在数据点原点
        ax.spines['left'].set_position(('data', 0))
        plt.plot(x, dy)
        plt.savefig('imgs/' + des + '_grad.png')


#SRelu tl, al, tr, ar

def single_parameter_test():
    funcs = [(prelu, d_prelu, 'prelu'),(rrelu, d_rrelu, 'rrelu'),(elu, d_elu, 'elu'), (isrlu, d_isrlu, 'isrlu'), (softexponential, d_softexponential, 'softexpinential')]
    for f in funcs:
        x = [i/10 for i in range(-100,100)]
        d_func = f[1]
        func = f[0]
        desc = f[2]
        print(desc)
        plt.figure()
        for a in range(0,300,30):
            a = a/100
            y = []
            dy = []
            for e in x:
                y.append(func(e, a))
            plt.plot(x, y)

        
        ax = plt.gca()    # 得到图像的Axes对象
        ax.spines['right'].set_color('none')   # 将图像右边的轴设为透明
        ax.spines['top'].set_color('none')     # 将图像上面的轴设为透明
        ax.xaxis.set_ticks_position('bottom')    # 将x轴刻度设在下面的坐标轴上
        ax.yaxis.set_ticks_position('left')         # 将y轴刻度设在左边的坐标轴上
        ax.spines['bottom'].set_position(('data', 0))   # 将两个坐标轴的位置设在数据点原点
        ax.spines['left'].set_position(('data', 0))
        
        plt.savefig('imgs/'+desc+'.png')   

        plt.figure()
        for a in range(0,300,30):
            a = a/100
            y = []
            dy = []
            for e in x:
                dy.append(d_func(e, a))
            plt.plot(x, dy)

        
        ax = plt.gca()    # 得到图像的Axes对象
        ax.spines['right'].set_color('none')   # 将图像右边的轴设为透明
        ax.spines['top'].set_color('none')     # 将图像上面的轴设为透明
        ax.xaxis.set_ticks_position('bottom')    # 将x轴刻度设在下面的坐标轴上
        ax.yaxis.set_ticks_position('left')         # 将y轴刻度设在左边的坐标轴上
        ax.spines['bottom'].set_position(('data', 0))   # 将两个坐标轴的位置设在数据点原点
        ax.spines['left'].set_position(('data', 0))
        
        plt.savefig('imgs/'+desc+'_grad.png')   

    
def draw_selu():
    des = 'selu'
    func = selu
    d_func = d_selu

    x = [i/10 for i in range(-100,100)]
    
        
    y = []
    dy = []
    for e in x:
        y.append(func(e, 1.67326,1.0507))
        dy.append(d_func(e, 1.67326,1.0507))

    plt.figure()
    ax = plt.gca()    # 得到图像的Axes对象
    ax.spines['right'].set_color('none')   # 将图像右边的轴设为透明
    ax.spines['top'].set_color('none')     # 将图像上面的轴设为透明
    ax.xaxis.set_ticks_position('bottom')    # 将x轴刻度设在下面的坐标轴上
    ax.yaxis.set_ticks_position('left')         # 将y轴刻度设在左边的坐标轴上
    ax.spines['bottom'].set_position(('data', 0))   # 将两个坐标轴的位置设在数据点原点
    ax.spines['left'].set_position(('data', 0))

    plt.plot(x, y)
    plt.savefig('imgs/' + des + '.png')

    plt.figure()
    ax = plt.gca()    # 得到图像的Axes对象
    ax.spines['right'].set_color('none')   # 将图像右边的轴设为透明
    ax.spines['top'].set_color('none')     # 将图像上面的轴设为透明
    ax.xaxis.set_ticks_position('bottom')    # 将x轴刻度设在下面的坐标轴上
    ax.yaxis.set_ticks_position('left')         # 将y轴刻度设在左边的坐标轴上
    ax.spines['bottom'].set_position(('data', 0))   # 将两个坐标轴的位置设在数据点原点
    ax.spines['left'].set_position(('data', 0))
    plt.plot(x, dy)
    plt.savefig('imgs/' + des + '_grad.png')

if __name__ == '__main__':
    #test_no_parameter()
    #single_parameter_test()
    draw_selu()