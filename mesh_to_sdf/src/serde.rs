use std::path::Path;

use super::*;
use ::serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Serialize signed distance fields struct.
#[derive(Serialize)]
#[serde(bound = "V: Serialize + DeserializeOwned")]
pub enum SerializeSdf<'a, V: Point> {
    /// Serialize a generic signed distance fields computed with `generate_sdf`.
    Generic(SerializeGeneric<'a, V>),
    /// Serialize a grid signed distance fields computed with `generate_grid_sdf`.
    Grid(SerializeGrid<'a, V>),
}

/// Serialize a generic signed distance fields computed with `generate_sdf`.
/// Should be used with `SerializeSdf::Generic`.
#[derive(Serialize)]
#[serde(bound = "V: Serialize + DeserializeOwned")]
pub struct SerializeGeneric<'a, V: Point> {
    /// Query points used to generate the signed distance field.
    pub query_points: &'a [V],
    /// Computed distances to the query points.
    pub distances: &'a [f32],
}

/// Serialize a grid signed distance fields computed with `generate_grid_sdf`.
/// Should be used with `SerializeSdf::Grid`.
#[derive(Serialize)]
#[serde(bound = "V: Serialize + DeserializeOwned")]
pub struct SerializeGrid<'a, V: Point> {
    /// Grid used to generate the signed distance field.
    pub grid: &'a Grid<V>,
    /// Computed distances to the grid cells.
    pub distances: &'a [f32],
}

/// Serialize a signed distance fields struct to a byte array.
fn serialize<V: Point + Serialize + DeserializeOwned>(sdf: &SerializeSdf<V>) -> Vec<u8> {
    rmp_serde::to_vec(sdf).unwrap()
}

/// Deserialize signed distance fields struct.
#[derive(Deserialize)]
#[serde(bound = "V: Serialize + DeserializeOwned")]
pub enum DeserializeSdf<V: Point> {
    /// Deserialized generic signed distance fields computed with `generate_sdf`.
    Generic(DeserializeGeneric<V>),
    /// Deserialized grid signed distance fields computed with `generate_grid_sdf`.
    Grid(DeserializeGrid<V>),
}

/// Deserializd generic signed distance fields computed with `generate_sdf`.
/// Should be used with `DeserializeSdf::Generic`.
#[derive(Deserialize)]
#[serde(bound = "V: Serialize + DeserializeOwned")]
pub struct DeserializeGeneric<V: Point> {
    /// Query points used to generate the signed distance field.
    pub query_points: Vec<V>,
    /// Computed distances to the query points.
    pub distances: Vec<f32>,
}

/// Deserialized grid signed distance fields computed with `generate_grid_sdf`.
/// Should be used with `DeserializeSdf::Grid`.
#[derive(Deserialize)]
#[serde(bound = "V: Serialize + DeserializeOwned")]
pub struct DeserializeGrid<V: Point> {
    /// Grid used to generate the signed distance field.
    pub grid: Grid<V>,
    /// Computed distances to the grid cells.
    pub distances: Vec<f32>,
}

/// Deserialize a byte array to a signed distance fields struct.
fn deserialize<V: Point + Serialize + DeserializeOwned>(data: &[u8]) -> DeserializeSdf<V> {
    rmp_serde::from_slice(data).unwrap()
}

/// Save a signed distance fields struct to a file.
///
/// ```no_run
/// use mesh_to_sdf::*;
/// let query_points = [cgmath::Vector3::new(0., 0., 0.)];
/// let distances = [1.];
/// let ser = SerializeSdf::Generic(SerializeGeneric {
///     query_points: &query_points,
///     distances: &distances,
/// });
/// let path = "path/to/sdf.bin";
/// save_to_file(&ser, path).expect("Failed to save sdf");
/// ```
pub fn save_to_file<V: Point + Serialize + DeserializeOwned, P: AsRef<Path>>(
    sdf: &SerializeSdf<V>,
    path: P,
) -> Result<(), std::io::Error> {
    std::fs::write(path, serialize(sdf))
}

/// Read a signed distance fields struct from a file.
/// You need to make sure the Point type is the same as the one used to serialize the data.
///
/// ```no_run
/// use mesh_to_sdf::*;
/// let path = "path/to/sdf.bin";
/// let deserialized = read_from_file::<cgmath::Vector3<f32>, _>(path).expect("Failed to read sdf");
/// match deserialized {
///     DeserializeSdf::Generic(DeserializeGeneric { query_points, distances }) => {
///         // ...
///     },
///     DeserializeSdf::Grid(DeserializeGrid { grid, distances }) => {
///         // ...
///     },
/// }
/// ```
pub fn read_from_file<V: Point + Serialize + DeserializeOwned, P: AsRef<Path>>(
    path: P,
) -> Result<DeserializeSdf<V>, std::io::Error> {
    Ok(deserialize(&std::fs::read(path)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::*;

    #[test]
    fn test_serde() {
        let queries = [
            cgmath::Vector3::new(1., 2., 3.),
            cgmath::Vector3::new(6., 5., 4.),
        ];
        let distances = [1.0, 3.0];
        let ser = SerializeSdf::Generic(SerializeGeneric {
            query_points: &queries,
            distances: &distances,
        });

        let data = serialize(&ser);
        let de: DeserializeSdf<cgmath::Vector3<f32>> = deserialize(&data);

        match (&ser, &de) {
            (SerializeSdf::Generic(ser), DeserializeSdf::Generic(de)) => {
                assert_eq!(ser.query_points, &de.query_points);
                assert_eq!(ser.distances, de.distances);
            }
            _ => panic!("Mismatch"),
        }
    }

    #[test]
    fn test_serde_grid() {
        let grid = Grid::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7, 8, 9]);
        let distances = (0..grid.get_total_cell_count())
            .map(|i| i as f32)
            .collect::<Vec<_>>();

        let ser = SerializeSdf::Grid(SerializeGrid {
            grid: &grid,
            distances: &distances,
        });

        let data = serialize(&ser);
        let de: DeserializeSdf<[f32; 3]> = deserialize(&data);

        match (&ser, &de) {
            (SerializeSdf::Grid(ser), DeserializeSdf::Grid(de)) => {
                assert_eq!(ser.grid, &de.grid);
                assert_eq!(ser.distances, de.distances);
            }
            _ => panic!("Mismatch"),
        }
    }

    #[test]
    fn test_serde_file() -> std::io::Result<()> {
        let dir = tempdir()?;
        let file_path = dir.path().join("sdf.bin");

        let queries = [
            cgmath::Vector3::new(1., 2., 3.),
            cgmath::Vector3::new(6., 5., 4.),
        ];
        let distances = [1.0, 3.0];
        let ser = SerializeSdf::Generic(SerializeGeneric {
            query_points: &queries,
            distances: &distances,
        });

        save_to_file(&ser, &file_path)?;

        let de: DeserializeSdf<cgmath::Vector3<f32>> = read_from_file(&file_path)?;

        match (&ser, &de) {
            (SerializeSdf::Generic(ser), DeserializeSdf::Generic(de)) => {
                assert_eq!(ser.query_points, &de.query_points);
                assert_eq!(ser.distances, de.distances);
            }
            _ => panic!("Mismatch"),
        }

        Ok(())
    }
}
