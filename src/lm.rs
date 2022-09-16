// Linear regression

type Data1D = Vec<(f64, f64)>;

pub struct LinearRegression1D<'a> {
    w: f64,
    b: f64,
    learning_rate: f64,
    epochs: usize,
    data: &'a Data1D,
}

impl<'a> LinearRegression1D<'a> {
    pub fn new(data: &'a Data1D) -> Self {
        LinearRegression1D {
            w: 1.0,
            b: 0.0,
            learning_rate: 0.001,
            epochs: 10000,
            data,
        }
    }

    pub fn train(&mut self) {
        let n = self.data.len() as f64;

        // Add adagrad or improvement stuff here

        for _ in 0..self.epochs {
            let mut dl_dw: f64 = 0.0;
            let mut dl_db: f64 = 0.0;

            // Update partial derivatives
            for i in 0..self.data.len() {
                let x1 = self.data[i].0;
                let y = self.data[i].1;
                let predicted_y = self.w * x1 + self.b;

                dl_dw += -2.0 * x1 * (y - predicted_y);
                dl_db += -2.0 * (y - predicted_y);
            }

            self.w -= (dl_dw / n) * self.learning_rate;
            self.b -= (dl_db / n) * self.learning_rate;
        }
    }

    pub fn params(&self) -> Vec<f64> {
        vec![self.w, self.b]
    }
}

#[cfg(test)]
mod tests {

    // Need to generate test data that will be consistent enough that tests are useful

    use super::*;

    #[test]
    fn basic_test() {
        let test_data = vec![(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)];
        let mut test_model = LinearRegression1D::new(&test_data);
        test_model.train();
        let test_w = test_model.params()[0];
        let test_b = test_model.params()[1];
        println!("w: {}, b: {}", &test_w, &test_b);
        assert!(test_w >= 0.99 && test_w <= 1.01);
        assert!(test_b >= 0.99 && test_b <= 1.01);
    }
}
