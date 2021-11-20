import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset


def generate_cvrptw_data(dataset_size, vrp_size):
    CAPACITIES = {
        10: 20.,
        25: 40.,
        50: 40.,
        100: 50.
    }
    start_time = np.random.uniform(0, vrp_size / 2, size=(dataset_size, vrp_size, 1))
    due_time = np.random.uniform(vrp_size / 2 + 1, vrp_size, size=(dataset_size, vrp_size, 1))

    return list(zip(
        np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
        np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
        np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
        np.full(dataset_size, CAPACITIES[vrp_size]).tolist(),  # Capacity, same for whole dataset
        np.concatenate((start_time, due_time), axis=-1).tolist(),  # Time window
        np.full(dataset_size, 100.0).tolist(),  # Due date
        np.full(dataset_size, 1.0).tolist()  # Service time, same for whole dataset
    ))


'''
read solomon dataset
order: depot, loc, demand, capacity, time_window, due, service_time
CUST NO.  XCOORD.   YCOORD.    DEMAND    READY TIME  DUE DATE   SERVICE  TIME
'''


def read_data(path: str) -> (int, int, list):
    filelist = os.listdir(path)
    depot_loc = []
    customer_loc = []
    customer_demand = []
    car_capacity = []
    customer_time_window = []
    due_date = []
    service_time = []

    for txt in filelist:
        txtpath = path + "/" + txt
        with open(txtpath, 'r', ) as f:
            lines = f.readlines()
        capacity = (float)(lines[4].split()[-1])
        max_vehicle = (int)(lines[4].split()[0])
        depot = [(float)(lines[9].split()[1]) / 100, (float)(lines[9].split()[2]) / 100]
        service = (float)(lines[10].split()[-1])
        due = (float)(lines[9].split()[5]) / service

        lines = lines[10:]
        loc = []
        demand = []
        tw = []
        for line in lines:
            info = [float(j) for j in line.split()]
            if len(info) == 7:
                node_loc = [info[1] / 100, info[2] / 100]
                loc.append(node_loc)

                node_demand = info[3] / 5
                demand.append(node_demand/capacity)

                node_tw = [info[4] / 50, info[5] / 50]
                tw.append(node_tw)

        car_capacity.append(capacity) # 单一车容量，可以考虑删除capacity

        depot_loc.append(depot)
        customer_loc.append(loc)
        customer_demand.append(demand)
        customer_time_window.append(tw)
        due_date.append(due)
        service_time.append(service)

    # start_time = np.random.uniform(0, vrp_size / 2, size=(dataset_size, vrp_size, 1))
    # due_time = np.random.uniform(vrp_size / 2 + 1, vrp_size, size=(dataset_size, vrp_size, 1))
    #
    # return list(zip(
    #     np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
    #     np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
    #     np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
    #     np.full(dataset_size, CAPACITIES[vrp_size]).tolist(),  # Capacity, same for whole dataset
    #     np.concatenate((start_time, due_time), axis=-1).tolist(),  # Time window
    #     np.full(dataset_size, 100.0).tolist(),  # Due date
    #     np.full(dataset_size, 1.0).tolist()  # Service time, same for whole dataset
    # ))

    print(len(filelist))

    return list(zip(
        depot_loc,  # Depot location
        customer_loc,  # Node locations
        customer_demand,  # Demand
        car_capacity,  # Capacity, same for whole dataset
        customer_time_window,  # Time window
        due_date,  # Due date
        service_time  # Service time, same for whole dataset
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[25, 50, 100],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")
    parser.add_argument('--read_exist_dataset', action='store_true', help='Not generating data but read data')
    parser.add_argument("--dataset_path", type=str, help="Existing datasetpath")

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    for graph_size in opts.graph_sizes:

        os.makedirs(opts.data_dir, exist_ok=True)

        if opts.filename is None:
            filename = os.path.join(opts.data_dir, "{}_{}_seed{}.pkl".format(
                graph_size, opts.name, opts.seed))
        else:
            filename = check_extension(opts.filename)

        assert opts.f or not os.path.isfile(check_extension(filename)), \
            "File already exists! Try running with -f option to overwrite."

        np.random.seed(opts.seed)
        if opts.read_exist_dataset:
            dataset = read_data(opts.dataset_path)
        else:
            dataset = generate_cvrptw_data(
                opts.dataset_size, graph_size)

        print(len(dataset))
        print(dataset[0])

        save_dataset(dataset, filename)
