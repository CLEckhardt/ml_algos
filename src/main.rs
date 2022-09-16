mod lm;
use crate::lm::LinearRegression1D;

fn read_csv(file: &str) -> Vec<(f64, f64)> {
    //
    let mut output: Vec<(f64, f64)> = Vec::new();
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(file)
        .unwrap();
    for result in reader.deserialize() {
        let record: (f64, f64) = result.unwrap();
        output.push(record);
    }

    output
}

fn main() {
    //
    let data = read_csv("test_1d_50.csv");
    let mut model = LinearRegression1D::new(&data);
    model.train();
    println!("w: {}, b: {}", model.params()[0], model.params()[1]);
}
