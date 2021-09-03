// Multilayer feed forward networks
// Convolutional Neural Networks,
// Recurrent Neural Networks
// LSTM

pub trait Activate {
    fn activate(&self, k: f64) -> f64;
}

#[derive(Clone)]
pub enum AFunc {
    Linear(LinearFunction),
}

impl Activate for AFunc {
    fn activate(&self, k: f64) -> f64 {
        match self {
            AFunc::Linear(f) => f.activate(k),
        }
    }
}

#[derive(Clone)]
pub enum RBFunc {}

impl Activate for RBFunc {
    fn activate(&self, k: f64) -> f64 {
        todo!()
    }
}

#[derive(Clone)]
pub struct LinearFunction {
    pub slope: f64,
    pub constant: f64,
}

impl Activate for LinearFunction {
    fn activate(&self, k: f64) -> f64 {
        (self.slope * k) + self.constant
    }
}

enum Input {
    F64(f64),
}

enum Output {
    F64(f64),
}

#[derive(Clone)]
pub struct Perceptron<F>
where
    F: Activate,
{
    n_iter: u128,
    weights: Vec<f64>,
    activate: F,
}

enum Layer<F>
where
    F: Activate,
{
    P { ps: Vec<Perceptron<F>> },
}

enum Nn {
    FF {
        hidden: Layer<AFunc>,
        p: Perceptron<AFunc>,
    },
    RBF {
        hidden: Layer<RBFunc>,
        p: Perceptron<RBFunc>,
    },
    DFF {
        hiddens: Vec<Layer<AFunc>>,
        p: Perceptron<AFunc>,
    },
}

//impl Perceptron {
//    pub fn new<W>(weights: Vec<W>, activate: AFunc) -> Self
//    where
//        W: Into<f64>,
//    {
//        let weights: Vec<f64> = weights.into_iter().map(Into::into).collect();
//
//        Perceptron { weights, activate }
//    }
//
//    pub fn compute(&mut self, xs: Arc<Vec<Node>>) -> f64 {
//        let k: f64 = 0.0;
//        for x in xs.into_iter()
//        for x in xs.into_iter() {}
//        let k: f64 = xs
//            .iter()
//            .map(|x| match x {
//                Input::F64(f) => *f,
//            })
//            .zip(self.weights.clone().into_iter())
//            .map(|(x, w)| x * w)
//            .sum();
//
//        self.activate.apply(k)
//    }
//}
//
//pub fn feed_forward(inputs: Vec<Layer>, mut nodes: Vec<Layer>) -> Vec<Node> {
//    for node in nodes.iter_mut() {
//        match node {
//            Node::P(p) => p.inputs = inputs.clone(),
//            Node::V(_) => (),
//        }
//    }
//    nodes
//}
//
//pub fn deep_feed_forward(mut inputs: Vec<Node>, deeps: Vec<Vec<Node>>) -> Vec<Node> {
//    for nodes in deeps.into_iter() {
//        inputs = feed_forward(inputs, nodes);
//    }
//}
