# Tutorial: Rust for Python
> A Tutorial on How to Call Rust Functions in Python

## Requirements

- Rust (-> [install](https://doc.rust-lang.org/cargo/getting-started/installation.html))
- Python >= 3.7 (-> [install](https://www.python.org/downloads/))

## Structure

We build a `lib` project for Rust that will compile and bind a function into a Python `wheel` with the help of the [PyO3](https://github.com/PyO3/pyo3) and [Maturin](https://maturin.rs/) tools.

### Example Application

Our example application is a simple [KNN implementation](./knn_rs/knn.py), both in _Python_ and _Rust_.

- PythonKNN
- RustKNN

Both implementations use the same amount of for loops and have a quite similar behavior. 
To compare them, the [benchmark](./knn_rs/benchmark.py) script runs both implementations on the [Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) and reports on both quality and performance results.

Both KNN versions and the benchmark script will be bundled in the `wheel` file. In fact, all resources in [./knn_rs](./knn_rs) will be available after installing the wheel. Maturin does this automatically. The folder needs to have the same name as the Rust project.

### Makefile

We provide a [Makefile](./Makefile) and a [tasks.sh](./tasks.sh) file with commands and functions to quickly generate the wheel.

### lib.rs

This file holds the code for the Rust version of the KNN code. [lib.rs](src/lib.rs) references the [python_bindings.rs](src/python_bindings.rs) file:

```rust
#[cfg(feature = "python")]
mod python_bindings;
```

This way, the [python_bindings](src/python_bindings.rs) are only compiled if we turn on the _python_ feature during compilation.

### python_bindings.rs

```rust
use ndarray::{ArrayView1, ArrayView2};
use crate::knn;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;


#[pyfunction]
fn knn_algorithm<'py>( // c)
    py: Python<'py>,
    x_py: PyReadonlyArray2<'py, f64>, // d)
    y_py: PyReadonlyArray1<'py, i64>,
    k: usize
) -> PyResult<&'py PyArray1<i64>> {
    let x: ArrayView2<f64> = x_py.as_array(); // e)
    let y: ArrayView1<i64> = y_py.as_array();

    let predicted = knn(x, y, k);
    Ok(predicted.into_pyarray(py)) // f)
}


#[pymodule] // a)
fn knn_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(knn_algorithm, m)?)?; // b)

    Ok(())
}
```

This file defines the functions that will be exposed to Python. In order to do so, we have to generate a `pymodule` (a) that is a function defining all the exposed functions.
In our case, we only have one of these definitions (b).
The exposed function `knn_algorithm` (c) takes in a `Python` object and the parameters that will be needed from the Python code that will call this function.
_Numpy_ arrays have to be transformed to an `ndarray`. They are received as a `PyReadonlyArray` (d) and then translated with a simple `as_array()` (e) function call.
Similarly, an `ndarray` must be transformed into a `PyArray` before returning it (f).

### knn.py

To use the exposed Rust function, we have to import it first in [knn.py](./knn_rs/knn.py). This will probably confuse the IDE, because it does not find the referenced module or function.

```python
from .knn_rs import knn_algorithm
```

Then, we can call the `knn_algorithm` function like any other Python function that we know.

```python
class RustKNN(KNN):
    def __init__(self, k: int = 3):
        self.k = k

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return knn_algorithm(X, y, self.k)
```

### Important to know

The generated `wheel` is compiled for your specific operating system only. If you want to generate a generic version (e.g. _manylinux_), it is easiest to work with Docker containers or toolchains for your target system ([more here](https://pyo3.rs/v0.8.1/building_and_distribution.html#cross-compiling)).

## Steps to Reproduce

### 0. Clone repo and create a new Python environment

```shell
git clone https://github.com/wenig/tutorial-rust-for-python.git
cd tutorial-rust-for-python
```

For example:
```shell
python -m venv rust_tutorial
source rust_tutorial/bin/activate
```

### 1. Test whether Cargo builds the Rust code correctly

```shell
cargo build --features python
```

### 2. Build and install `knn_rs`

```shell
make install
```

or

```shell
pip install -r requirements.txt
bash ./tasks.sh release-install
```

### 3. Change directory

Sometimes Python is confused that there is a folder `knn_rs` named like a library that we want to import and prefers the folder over the installed library. That's why we need to go somewhere else.

```shell
cd ..
```

### 4. Benchmark implementations

```shell
python
```

```python
from knn_rs.benchmark import benchmark
benchmark()
```
