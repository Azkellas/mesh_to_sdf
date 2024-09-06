use std::path::Path;

use super::*;
use ::serde::{de::DeserializeOwned, Deserialize, Serialize};

// ----------------------------------------------------------------------------
// Serde Error type

/// Error type for serialization and deserialization.
#[derive(Debug)]
pub enum SerdeError {
    /// Failed to deserialize the data via rmp-serde.
    SerializationFailed,
    /// Failed to deserialize or deserialize the data via rmp-serde.
    DeserializationFailed,
    /// Failed to read or write the file.
    IoError(std::io::Error),
}

impl From<std::io::Error> for SerdeError {
    fn from(e: std::io::Error) -> Self {
        SerdeError::IoError(e)
    }
}

impl From<rmp_serde::encode::Error> for SerdeError {
    fn from(_: rmp_serde::encode::Error) -> Self {
        SerdeError::SerializationFailed
    }
}

impl From<rmp_serde::decode::Error> for SerdeError {
    fn from(_: rmp_serde::decode::Error) -> Self {
        SerdeError::DeserializationFailed
    }
}

// ----------------------------------------------------------------------------
// Serialization structs

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

/// Version of the serialization format.
/// This is used to ensure backward compatibility when deserializing.
#[derive(Serialize)]
#[serde(bound = "V: Serialize + DeserializeOwned")]
enum SerializeVersion<'a, V: Point> {
    V1(&'a SerializeSdf<'a, V>),
}

// ----------------------------------------------------------------------------
// Deserialization structs

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

/// Version of the deserialization format.
/// This is used to ensure backward compatibility when deserializing.
#[derive(Deserialize)]
#[serde(bound = "V: Serialize + DeserializeOwned")]
enum DeserializeVersion<V: Point> {
    V1(DeserializeSdf<V>),
}

// ----------------------------------------------------------------------------
// Functions

/// Serialize a signed distance fields struct to a byte array.
fn serialize<V: Point + Serialize + DeserializeOwned>(
    sdf: &SerializeSdf<V>,
) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    // Serialize using the latest version
    rmp_serde::to_vec(&SerializeVersion::V1(sdf))
}

/// Deserialize a byte array to a signed distance fields struct.
fn deserialize<V: Point + Serialize + DeserializeOwned>(
    data: &[u8],
) -> Result<DeserializeSdf<V>, SerdeError> {
    let versioned_sdf: DeserializeVersion<V> = rmp_serde::from_slice(data)?;
    Ok(match versioned_sdf {
        DeserializeVersion::V1(sdf) => sdf,
    })
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
) -> Result<(), SerdeError> {
    std::fs::write(path, serialize(sdf)?)?;
    Ok(())
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
) -> Result<DeserializeSdf<V>, SerdeError> {
    deserialize(&std::fs::read(path)?)
}

// ----------------------------------------------------------------------------
// Tests

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::*;

    #[test]
    fn test_serde() -> Result<(), SerdeError> {
        let queries = [
            cgmath::Vector3::new(1., 2., 3.),
            cgmath::Vector3::new(6., 5., 4.),
        ];
        let distances = [1.0, 3.0];
        let ser = SerializeSdf::Generic(SerializeGeneric {
            query_points: &queries,
            distances: &distances,
        });

        let data = serialize(&ser)?;
        let de: DeserializeSdf<cgmath::Vector3<f32>> = deserialize(&data)?;

        match (&ser, &de) {
            (SerializeSdf::Generic(ser), DeserializeSdf::Generic(de)) => {
                assert_eq!(ser.query_points, &de.query_points);
                assert_eq!(ser.distances, de.distances);
            }
            _ => panic!("Mismatch"),
        }

        Ok(())
    }

    #[test]
    fn test_serde_grid() -> Result<(), SerdeError> {
        let grid = Grid::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7, 8, 9]);
        let distances = (0..grid.get_total_cell_count())
            .map(|i| i as f32)
            .collect::<Vec<_>>();

        let ser = SerializeSdf::Grid(SerializeGrid {
            grid: &grid,
            distances: &distances,
        });

        let data = serialize(&ser)?;
        let de: DeserializeSdf<[f32; 3]> = deserialize(&data)?;

        match (&ser, &de) {
            (SerializeSdf::Grid(ser), DeserializeSdf::Grid(de)) => {
                assert_eq!(ser.grid, &de.grid);
                assert_eq!(ser.distances, de.distances);
            }
            _ => panic!("Mismatch"),
        }

        Ok(())
    }

    #[test]
    fn test_serde_file() -> Result<(), SerdeError> {
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

    #[test]
    fn test_backward_compatibility_serde_generic_v1() -> Result<(), SerdeError> {
        let queries = [
            cgmath::Vector3::new(1., 2., 3.),
            cgmath::Vector3::new(6., 5., 4.),
        ];
        let distances = [1.0, 3.0];
        let ser = SerializeSdf::Generic(SerializeGeneric {
            query_points: &queries,
            distances: &distances,
        });

        let path = "tests/sdf_generic_v1.bin";

        // This was done with the version V1 of the serialization format
        // save_to_file(&ser, path);

        // Now we make sure we can read it with the current version
        let de: DeserializeSdf<cgmath::Vector3<f32>> = read_from_file(path)?;

        match (&ser, &de) {
            (SerializeSdf::Generic(ser), DeserializeSdf::Generic(de)) => {
                assert_eq!(ser.query_points, &de.query_points);
                assert_eq!(ser.distances, de.distances);
            }
            _ => panic!("Mismatch"),
        }

        Ok(())
    }

    #[test]
    fn test_backward_compatibility_serde_grid_v1() -> Result<(), SerdeError> {
        let grid = Grid::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7, 8, 9]);
        let distances = (0..grid.get_total_cell_count())
            .map(|i| i as f32)
            .collect::<Vec<_>>();

        let ser = SerializeSdf::Grid(SerializeGrid {
            grid: &grid,
            distances: &distances,
        });

        let path = "tests/sdf_grid_v1.bin";

        // This was done with the version V1 of the serialization format
        // save_to_file(&ser, path)?;

        // Now we make sure we can read it with the current version
        let de: DeserializeSdf<[f32; 3]> = read_from_file(path)?;

        match (&ser, &de) {
            (SerializeSdf::Grid(ser), DeserializeSdf::Grid(de)) => {
                assert_eq!(ser.grid, &de.grid);
                assert_eq!(ser.distances, de.distances);
            }
            _ => panic!("Mismatch"),
        }

        Ok(())
    }
}
