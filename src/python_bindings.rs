use ndarray::{ArrayView1, ArrayView2};
use crate::knn;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;


#[pyfunction]
fn knn_algorithm<'py>(
    py: Python<'py>,
    x_py: PyReadonlyArray2<'py, f64>,
    y_py: PyReadonlyArray1<'py, i64>,
    k: usize
) -> PyResult<&'py PyArray1<i64>> {
    let x: ArrayView2<f64> = x_py.as_array();
    let y: ArrayView1<i64> = y_py.as_array();

    let predicted = knn(x, y, k);
    Ok(predicted.into_pyarray(py))
}


#[pymodule]
fn knn_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(knn_algorithm, m)?)?;

    Ok(())
}
