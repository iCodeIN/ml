// Multilayer feed forward networks
// Convolutional Neural Networks,
// Recurrent Neural Networks
// LSTM

use std::{
    iter::Sum,
    marker::PhantomData,
    ops::{Add, Mul},
};

pub struct Perceptron<X, W, F, O>
where
    W: Clone + Mul<X>,
    <W as Mul<X>>::Output: Add,
    <<W as Mul<X>>::Output as Add>::Output: Sum<<W as Mul<X>>::Output>,
    F: Fn(<<W as Mul<X>>::Output as Add>::Output) -> O,
{
    afunc: F,
    weights: Vec<W>,
    _k: PhantomData<X>,
}

impl<X, W, F, O> Perceptron<X, W, F, O>
where
    W: Clone + Mul<X>,
    <W as Mul<X>>::Output: Add,
    <<W as Mul<X>>::Output as Add>::Output: Sum<<W as Mul<X>>::Output>,
    F: Fn(<<W as Mul<X>>::Output as Add>::Output) -> O,
{
    pub fn new(afunc: F, weights: Vec<W>) -> Self {
        Perceptron {
            afunc,
            weights,
            _k: PhantomData,
        }
    }

    pub fn apply(&self, xs: Vec<X>) -> O {
        let k: <<W as Mul<X>>::Output as Add>::Output = xs
            .into_iter()
            .zip(self.weights.clone().into_iter())
            .map(|(x, w)| w * x)
            .sum();

        (self.afunc)(k)
    }
}
