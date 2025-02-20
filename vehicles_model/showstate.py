import matplotlib.pyplot as plt

#matplotlib inline

from IPython import display

#可视化函数：

def show_state(env, step=0, info=""):

    plt.figure(3)

    plt.clf()

    plt.imshow(env.render(mode='rgb_array'))

    plt.title("Step: %d %s" % (step, info))

    plt.axis('off')

    display.clear_output(wait=True)

    display.display(plt.gcf())

