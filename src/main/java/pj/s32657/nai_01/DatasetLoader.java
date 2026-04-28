package pj.s32657.nai_01;

import java.io.*;
import java.util.*;

public class DatasetLoader {
  private final InputStream source;

  public DatasetLoader() {
    this.source = DatasetLoader.class.getResourceAsStream("/iris.csv");
  }

  public DatasetLoader(String path) throws FileNotFoundException {
    File file = new File(path);
    if (!file.exists())
      throw new FileNotFoundException("Dataset file not found: " + path);
    this.source = new FileInputStream(file);
  }

  private List<String[]> readRows() throws IOException {
    List<String[]> rows = new ArrayList<>();

    try (BufferedReader reader = new BufferedReader(new InputStreamReader(source))) {
      reader.readLine(); // skip header

      String line;
      int lineNumber = 1;
      while ((line = reader.readLine()) != null) {
        lineNumber++;
        String[] parts = line.split(",");
        if (parts.length < 2)
          throw new IOException("Invalid CSV format at line " + lineNumber + ": " + line);
        if (parts[parts.length - 1].trim().isEmpty())
          throw new IOException("Missing category at line " + lineNumber + ": " + line);
        rows.add(parts);
      }
    }

    return rows;
  }

  public Vector[] load() throws IOException {
    List<Vector> vectors = new ArrayList<>();

    for (String[] parts : readRows()) {
      double[] data = new double[parts.length - 1];
      try {
        for (int i = 0; i < data.length; i++)
          data[i] = Double.parseDouble(parts[i]);
      } catch (NumberFormatException e) {
        throw new IOException("Invalid numeric value: " + Arrays.toString(parts), e);
      }
      vectors.add(new Vector(data, parts[parts.length - 1].trim()));
    }

    return vectors.toArray(new Vector[0]);
  }

  public Observation[] loadObservations() throws IOException {
    List<Observation> observations = new ArrayList<>();

    for (String[] parts : readRows()) {
      Observation obs = new Observation(
          Arrays.copyOf(parts, parts.length - 1),
          parts[parts.length - 1].trim()
      );
      observations.add(obs);
    }

    return observations.toArray(new Observation[0]);
  }
}