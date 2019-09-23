import click
import numpy as np


@click.command()
@click.option("-n", "--count", "vector_count", default=500, help="Number of vectors.")
@click.option("-m", "--length", "vector_length", default=10, help="Length of each vector.")
@click.option("-o", "--output", "output_filename", default="vectors.csv", help="Output file name.")
def generate_test_data(vector_count=500, vector_length=10, output_filename="vectors.csv"):
    """
        Generates test data & saves it in `output_filename`.

        Test data is `vector_count` vectors in `vector_length` dimensions distributed uniformly
        in [-1, 1].
    """
    test_data = np.random.uniform(-1, 1, size=(vector_count, vector_length))
    np.savetxt(output_filename, test_data, delimiter=",")


if __name__ == "__main__":
    generate_test_data()
