import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import os

""""""
style.use('fivethirtyeight')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


def animate(i):
    path = 'C:/Research/WVU/Humen Bridge/Humen Bridge/Translation'
    path = os.path.join(path, '1_point.txt')
    graph_data = open(path, 'r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y))
    ax.clear()
    ax.plot(xs)
    # ax1.plot(xs,ys)


if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()
