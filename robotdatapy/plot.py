import matplotlib.pyplot as plt

def plot_pose_2d(pose, axes='xy', axis_len=1.0, ax=None):
    assert len(axes) == 2, "axes must be a string of length 2"
    ax_idx = []
    for i in range(2):
        if 'x' == axes[i]:
            ax_idx.append(0)
        elif 'y' == axes[i]:
            ax_idx.append(1)
        elif 'z' == axes[i]:
            ax_idx.append(2)
        else:
            assert False, "axes must be a string of x, y, or z"

    if ax is None:
        ax = plt.gca()
    for rob_ax, color in zip([0, 1, 2], ['red', 'green', 'blue']):
        ax.plot([pose[ax_idx[0],3], pose[ax_idx[0],3] + axis_len*pose[ax_idx[0],rob_ax]],
                [pose[ax_idx[1],3], pose[ax_idx[1],3] + axis_len*pose[ax_idx[1],rob_ax]], color=color)