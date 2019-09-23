import click
import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(a, b):
    """
        Calculates euclidean distances between `a` and `b`.
    """
    displacement = a - b
    squared_distance = np.sum(displacement ** 2)

    return np.sqrt(squared_distance)


def get_min_max_distances(data):
    """
        Calculates minimal & maximal distances between pair of vectors.
        Returns distances with corresponding index pairs.
    """
    vector_count, _ = data.shape

    min_distance = None
    min_pair = (None, None)

    max_distance = None
    max_pair = (None, None)

    for i in range(vector_count):
        first_vector = data[i]

        for j in range(i + 1, vector_count):
            second_vector = data[j]

            pair = (i, j)
            distance = euclidean_distance(first_vector, second_vector)

            if min_distance is None or min_distance > distance:
                min_distance = distance
                min_pair = pair

            if max_distance is None or max_distance < distance:
                max_distance = distance
                max_pair = pair

    return (min_distance, min_pair), (max_distance, max_pair)


def plot_histogram(data, min_distance, max_distance, filename):
    """
        Plots histogram of `data` to `filename` with 0.1 step.
    """
    vector_count, _ = data.shape

    histogram_step = 0.1

    data_length = max_distance - min_distance

    number_of_bins = int(data_length / histogram_step)

    distances_count = np.zeros(number_of_bins + 1, dtype=np.int32)

    for i in range(vector_count):
        first_vector = data[i]

        for j in range(i + 1, vector_count):
            second_vector = data[j]

            distance = euclidean_distance(first_vector, second_vector)

            distance_offset = distance - min_distance
            bin_number = int(distance_offset / histogram_step)

            distances_count[bin_number] += 1

    fig = plt.figure()

    xs = np.linspace(min_distance, max_distance, number_of_bins + 1)
    plt.bar(xs, distances_count)

    fig.savefig(filename)


@click.command()
@click.option("-i", "--input", "input_filename", default="vectors.csv", help="Input file name.")
@click.option("-o", "--output", "output_histogram_filename", default="histogram.png", help="Input file name.")
def main(input_filename="vectors.csv", output_histogram_filename="histogram.png"):
    """
        Main program. Data coming from `input_filename`.

        Finds pairs of vectors with minimal & maximal euclidean distances.
        Plots histogram to `output_histogram_filename`.
    """
    data = np.genfromtxt(input_filename, delimiter=',')

    (min_distance, min_pair), (max_distance, max_pair) = get_min_max_distances(data)

    print(f'Minimal distance: {min_distance} at {min_pair}')
    print(f'Maximal distance: {max_distance} at {max_pair}')

    plot_histogram(data, min_distance, max_distance, output_histogram_filename)


if __name__ == "__main__":
    main()
