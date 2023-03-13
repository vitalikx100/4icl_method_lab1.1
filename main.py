import math

import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate

def task1():
    A = np.random.randint(-3, 3, (5, 5))
    print(A)
    B=A.transpose()
    print(B)
    C=np.linalg.det(B)
    print(C)

def task2():
    B = np.array([2, 4, 3], int)
    C = np.array([[1, 2, 0], [1, 2, 1], [1, 1, 1]], int)
    D = np.dot(B, C)
    print('B =', B)
    print('C =', C)
    print('B * C =', D)


def task3():
    A = np.array([[-7, -5, -5], [0, 3, 0], [10, 5, 8]], int)
    print(A)
    w, v = np.linalg.eig(A)
    print("Собственные значения: ", w)
    print("Собственные векторы:\n ", v)


def task4():
    A = sp.integrate.quad(lambda x: math.exp(2*x) * math.cos(x), 0, math.pi/2)
    print(A)

def task5():
    A = sp.integrate.quad(lambda x: (1/(math.pow(x, 2) + 4*x + 9)), np.Inf, -np.Inf)
    print(A)

def task6():
    plt.figure(figsize=(12, 8), dpi=80)
    ax = plt.subplot()

    # удаляем правую и верхнюю прямоугольные границы
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Установить направление данных на координатной оси
    # 0 согласуется с нашей общей декартовой системой координат, 1 - противоположность
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    # Подготовить данные, использовать распаковку последовательности
    X = np.linspace(-4.999, 5, 256, endpoint=True)
    Y1 = numpy.log(X) + 2
    Y2 = -3 * X

    plt.plot(X, Y1, color="blue", linewidth=2.5, linestyle="-", label="ln(x) + 2")
    plt.plot(X, Y2, color="red", linewidth=2.5, linestyle="-", label="-3x")

    plt.xlim(X.min(), X.max()) # границы

    plt.xticks([-6, -5, -4, -3, -2, -1, +1, +2, +3, +4],
               [r'$-6$', r'$-5$', r'$-4$', r'$-3$', r'$-2$', r'$-1$', r'$+1$', r'$+2$', r'$+3$', r'$+4$'])

    # Усталавливаем границы оси OY
    plt.ylim(Y1.min(), Y1.max())
    # подписываем ось OY
    plt.yticks([-6, -5, -4, -3, -2, -1, 0, +1, +2, +3, +4, +5],
               [r'$-6$', r'$-5$', r'$-4$', r'$-3$', r'$-2$', r'$-1$', r'$0$', r'$+1$', r'$+2$', r'$+3$', r'$+4$',
                r'$+5$'])

    plt.box()

    plt.legend(loc='best', frameon=False, fontsize=14)  # положение легенды, сама легенда
    plt.grid()  # добавление сетки
    idx = np.argwhere(np.isclose(Y1, Y2, atol=0.1)).reshape(-1)
    ax.plot(X[idx], Y2[idx], color='red', linewidth=2.5, marker='o')
    plt.show()  # вывод графика



if __name__ == '__main__':
    task6()























