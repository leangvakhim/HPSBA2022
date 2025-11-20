import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class wsn:
    def __init__(self, width, height, num_nodes, r_sensing, resolution):
        self.width = width
        self.height = height
        self.num_nodes = num_nodes
        self.r_sensing = r_sensing

        x = np.arange(0, width + resolution, resolution)
        y = np.arange(0, height + resolution, resolution)
        self.grid_x, self.grid_y = np.meshgrid(x, y)


        self.target_points = np.column_stack((self.grid_x.ravel(), self.grid_y.ravel()))
        self.total_points = self.target_points.shape[0]

    def evaluate(self, positions_flat):
        sensors = positions_flat.reshape((self.num_nodes, 2))

        is_covered = np.zeros(self.total_points, dtype=bool)

        for sensor in sensors:
            dx = self.target_points[:, 0] - sensor[0]
            dy = self.target_points[:, 1] - sensor[1]
            dist_sq = dx**2 + dy**2

            is_covered |= (dist_sq <= self.r_sensing**2)

        coverage_count = np.sum(is_covered)
        coverage_ratio = coverage_count / self.total_points

        return 1.0 - coverage_ratio

    def plot(self, best_solution, title="WSN Node Coverage"):
        sensors = best_solution.reshape((self.num_nodes, 2))

        fig, ax = plt.subplots(figsize=(8, 8))

        for sensor in sensors:
            circle = patches.Circle((sensor[0], sensor[1]), self.r_sensing,
                                     alpha=0.3, fc='purple', ec='black')
            ax.add_patch(circle)
            ax.plot(sensor[0], sensor[1], 'k.', markersize=5)

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_title(title)
        ax.set_xlabel('Width (m)')
        ax.set_ylabel('Height (m)')
        ax.grid(True, linestyle='--')
        ax.set_aspect('equal')
        plt.show()


