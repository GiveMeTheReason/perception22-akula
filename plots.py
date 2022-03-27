from matplotlib import pyplot as plt


def plot_trajectory(trajectory, label):

    if trajectory.shape[1] > 3:
        trajectory = trajectory.T

    if trajectory.shape[1] in (2, 3):
        plt.plot(trajectory[:, 0], trajectory[:, 1], label = label)

    else:
        raise NotImplementedError()

    return None
