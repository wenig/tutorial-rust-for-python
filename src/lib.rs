#[cfg(feature = "python")]
mod python_bindings;

use std::collections::HashMap;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

fn knn(x: ArrayView2<f64>, y: ArrayView1<i64>, k: usize) -> Array1<i64> {
    let predicted: Vec<i64> = x.axis_iter(Axis(0)).map(|x1| {
        let mut closest: Vec<(f64, i64)> = Vec::with_capacity(k);

        for (i, x2) in x.axis_iter(Axis(0)).enumerate() {
            let distance = euclidean(x1, x2);

            if closest.len() < k {
                closest.push((distance, y[i]));
            } else if distance.lt(&closest_max(&closest)) {
                let max_position = argmax(&closest);
                closest[max_position] = (distance, y[i]);
            }
        }

        max_count(&closest)
    }).collect();

    Array1::from_iter(predicted)
}

fn euclidean(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter())
        .map(|(a_, b_)| ((*a_) - (*b_)) * ((*a_) - (*b_)))
        .fold(0., std::ops::Add::add)
        .sqrt()
}

fn closest_max(closest: &Vec<(f64, i64)>) -> f64 {
    closest.iter().map(|x| x.0).reduce(|a, b| a.max(b)).unwrap()
}

fn max_count(closest: &Vec<(f64, i64)>) -> i64 {
    let mut counter: HashMap<i64, i64> = HashMap::new();

    for (_, l) in closest.iter() {
        if counter.contains_key(l) {
            *(counter.get_mut(l).unwrap()) += 1;
        } else {
            counter.insert(*l, 1);
        }
    }

    counter.into_iter().max_by_key(|(_, v)| *v).unwrap().0
}

fn argmax(list: &Vec<(f64, i64)>) -> usize {
    let (position, _) = list.iter().enumerate().max_by(|(_, (a, _)), (_, (b, _))| a.partial_cmp(b).unwrap()).unwrap();
    position
}
