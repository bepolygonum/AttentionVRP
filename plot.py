import os
import numpy as np
import torch
from torch.utils.data import DataLoader


from matplotlib import pyplot as plt

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py
from utils.functions import load_model
from vrp.problem_vrp import CVRPTW


def discrete_cmap(N, base_cmap=None):
    """
      Create an N-bin discrete colormap from the specified input map
      """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def plot_vehicle_routes(data, route, ax1, markersize=5, visualize_demands=True, demand_scale=1, round_demand=False):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    """
    # print(data['loc'])
    # route is one sequence, separating different routes with 0 (depot)
    routes = [r[r != 0] for r in np.split(route.cpu().numpy(), np.where(route == 0)[0]) if (r != 0).any()]

    depot = data['depot'].cpu().numpy()
    locs = data['loc'].cpu().numpy()
    demands = data['demand'].cpu().numpy() * demand_scale
    capacity = demand_scale  # Capacity is always 1

    x_dep, y_dep = depot
    ax1.plot(x_dep, y_dep, 'sk', markersize=markersize * 4)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # legend = ax1.legend(loc='upper center')

    cmap = discrete_cmap(len(routes) + 2, 'nipy_spectral')
    dem_rects = []
    used_rects = []
    cap_rects = []
    qvs = []
    total_dist = 0
    for veh_number, r in enumerate(routes):
        # print(r)
        color = cmap(len(routes) - veh_number)  # Invert to have in rainbow order
        route_demands = demands[r - 1]
        coords = locs[r - 1, :]
        coordx = np.vstack((depot,coords,depot))
        xs, ys = coordx.transpose()
        total_route_demand = sum(route_demands)
        assert total_route_demand <= capacity + 1
        if not visualize_demands:
            ax1.plot(xs, ys, 'o', mfc=color, markersize=markersize, markeredgewidth=0.0)

        dist = 0
        x_prev, y_prev = x_dep, y_dep
        cum_demand = 0
        for (x, y), d in zip(coords, route_demands):
            dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)
            # print(dist)
            cap_rects.append(Rectangle((x, y), 0.01, 0.1))
            used_rects.append(Rectangle((x, y), 0.01, 0.1 * total_route_demand / capacity))
            dem_rects.append(Rectangle((x, y + 0.1 * cum_demand / capacity), 0.01, 0.1 * d / capacity))

            x_prev, y_prev = x, y
            cum_demand += d

        dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
        # print(dist)
        total_dist += dist
        qv = ax1.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units='xy',
            angles='xy',
            scale=1,
            color=color,
            label='R{}, # {}, c {} / {}, d {:.2f}'.format(
                veh_number,
                len(r),
                int(total_route_demand) if round_demand else total_route_demand,
                int(capacity) if round_demand else capacity,
                dist
            )
        )

        qvs.append(qv)

    ax1.set_title('{} routes, total distance {:.2f}'.format(len(routes), total_dist))
    ax1.legend(handles=qvs)

    pc_cap = PatchCollection(cap_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray')
    pc_used = PatchCollection(used_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray')
    pc_dem = PatchCollection(dem_rects, facecolor='black', alpha=1.0, edgecolor='black')

    if visualize_demands:
        ax1.add_collection(pc_cap)
        ax1.add_collection(pc_used)
        ax1.add_collection(pc_dem)
    return total_dist


if __name__ == "__main__":
    # load the model and dataset
    torch.manual_seed(1235)
    data_path = 'data\\25_test_seed1234.pkl'
    dataset = CVRPTW.make_dataset(data_path,size=25, num_samples=50)
    # Need a data loader to batch instances
    data_loader = DataLoader(dataset, batch_size=50)
    batch = next(iter(data_loader))
    model, _ = load_model('outputs/25/cvrptw25_rollout_20211116T191548/0epoch-99.pt')

    # Run the model
    model.eval()
    model.set_decode_type('greedy')
    with torch.no_grad():
        length, log_p, pi, time = model(batch, return_pi=True)
    tours = pi
    times = time
    # Plot the results
    for i, (data, tour, t) in enumerate(zip(dataset, tours, times)):
        fig, ax = plt.subplots(figsize=(10, 10))
        total_dist = plot_vehicle_routes(data, tour, ax, visualize_demands=True,
                            demand_scale=200, round_demand=True)
        fig.savefig(os.path.join('0', '{}_len{:.3}.png'.format(i,total_dist)))


