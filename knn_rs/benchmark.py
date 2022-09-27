from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import timeit

from .knn import PythonKNN, RustKNN


def benchmark():
    X, y = load_iris(return_X_y=True)

    seconds = timeit.timeit(lambda: PythonKNN().fit_transform(X, y), number=10)
    predicted = PythonKNN().fit_transform(X, y)
    print(f"PythonKNN: {accuracy_score(y, predicted)} in {seconds} seconds")

    seconds = timeit.timeit(lambda: RustKNN().fit_transform(X, y), number=10)
    predicted = RustKNN().fit_transform(X, y)
    print(f"RustKNN: {accuracy_score(y, predicted)} in {seconds} seconds")
