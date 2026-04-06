package pj.s32657.nai_01;

import java.util.Arrays;
import java.util.Map;
import java.util.Scanner;

public class UserInterface {
  private Vector[] dataset;
  private KNearestNeighbours knn;
  private Perceptron perceptron;
  private SplitDataset perceptronSets;
  private final Scanner scanner = new Scanner(System.in);

  public UserInterface() {
    DatasetLoader loader = new DatasetLoader();

    try { dataset = loader.load(); }
    catch (Exception e) {
      e.printStackTrace();
      return;
    }
  }

  public void run() {
    while (true) {
      System.out.println("\n1. KNN");
      System.out.println("2. Perceptron");
      System.out.println("0. Exit");
      System.out.print("> ");

      switch (scanner.nextLine().trim()) {
        case "1" -> knn();
        case "2" -> perceptron();
        case "0" -> { return; }
        default  -> System.out.println("Invalid option.");
      }
    }
  }

  private void knn() {
    while (true) {
      System.out.println("\n-- KNN --");
      System.out.println("1. Load dataset from file");
      System.out.println("2. Validate dataset accuracy");
      System.out.println("3. Classify a vector");
      System.out.println("4. Compare accuracy across k values");
      System.out.println("0. Back");
      System.out.print("> ");

      switch (scanner.nextLine().trim()) {
        case "1" -> loadDataset();
        case "2" -> validateAccuracy();
        case "3" -> classifyVector();
        case "4" -> compareK();
        case "0" -> { return; }
        default  -> System.out.println("Invalid option.");
      }
    }
  }

  private void loadDataset() {
    System.out.print("Path to CSV file: ");
    String path = scanner.nextLine().trim();

    try {
      dataset = new DatasetLoader(path).load();
      knn = new KNearestNeighbours(dataset);
      System.out.println("Loaded " + dataset.length + " vectors.");
    }

    catch (Exception e) {
      System.err.println("Error: " + e.getMessage());
    }
  }

  private void classifyVector() {
    System.out.print("Enter vector values separated by commas: ");
    String input = scanner.nextLine().trim();

    try {
      String[] parts = input.split(",");
      double[] data = new double[parts.length];
      for (int i = 0; i < parts.length; i++)
        data[i] = Double.parseDouble(parts[i].trim());

      String result = knn.categorize(new Vector(data));
      System.out.println("Classified as: " + result.toUpperCase());
    }

    catch (NumberFormatException e) {
      System.err.println("Invalid input: values must be numeric.");
    }

    catch (Exception e) {
      System.err.println("Error: " + e.getMessage());
    }
  }

  private int promptInt(String prompt, int defaultValue) {
    System.out.print(prompt);
    String input = scanner.nextLine().trim();
    if (input.isEmpty()) return defaultValue;
    try { return Integer.parseInt(input); }
    catch (NumberFormatException e) { return defaultValue; }
  }

  private void validateAccuracy() {
    SplitDataset sets = new PrepareDataset().trainTestSplit(dataset);
    double accuracy = EvaluationMetrics.measureAccuracy(sets);
    System.out.printf("Accuracy: %.2f%%%n", accuracy);
  }

  private void compareK() {
    int maxK = promptInt("Max k [default 5]: ", 5);
    int runs = promptInt("Runs per k [default 10]: ", 10);

    PrepareDataset prep = new PrepareDataset();

    System.out.println();
    for (int k = 1; k <= maxK; k++) {
      double sum = 0;
      for (int i = 0; i < runs; i++) {
        SplitDataset sets = prep.trainTestSplit(dataset);
        sum += EvaluationMetrics.measureAccuracy(sets, k);
      }

      System.out.printf("k=%-2d -> %.2f%%%n", k, sum / runs);
    }
  }

  private void perceptron() {
    while (true) {
      System.out.println("\n-- Perceptron --");
      System.out.println("1. Train");
      System.out.println("2. Predict");
      System.out.println("3. Plot hyperplane");
      System.out.println("4. Plot all hyperplanes");
      System.out.println("0. Back");
      System.out.print("> ");

      switch (scanner.nextLine().trim()) {
        case "1" -> trainPerceptron();
        case "2" -> predictPerceptron();
        case "3" -> plot();
        case "4" -> plotAll();
        case "0" -> { return; }
        default  -> System.out.println("Invalid option.");
      }
    }
  }

  private void trainPerceptron() {
    perceptronSets = new PrepareDataset().trainTestSplit(
      dataset, 0.7,
      x -> !x.category.equals("virginica"),
      true // shuffle
    );

//    perceptron = new Perceptron(dataset[0].size, 0.25, 0, 0.05);
    perceptron = new Perceptron(dataset[0].size, 5, 0, 0.01);
    var mapper = Map.of("versicolor", 0, "setosa", 1);
    perceptron.train(perceptronSets, mapper, 100, true);
  }

  private void predictPerceptron() {
    System.out.print("Enter 4 feature values separated by commas: ");
    String input = scanner.nextLine().trim();

    try {
      String[] parts = input.split(",");
      double[] data = new double[parts.length];
      for (int i = 0; i < parts.length; i++)
        data[i] = Double.parseDouble(parts[i].trim());

      int result = perceptron.predict(new Vector(data));
      System.out.println("Classified as: " + (result == 1 ? "setosa" : "versicolor"));
    }

    catch (NumberFormatException e) {
      System.err.println("Invalid input: values must be numeric.");
    }
  }

  private void plot() {
    if (perceptron == null || perceptronSets == null) {
      System.out.println("Train the perceptron first.");
      return;
    }

    int max = perceptron.dimension - 1;
    int fi0, fi1;

    while (true) {
      System.out.printf("Enter two feature indices (0-%d) separated by comma [default 1,2]: ", max);
      String input = scanner.nextLine().trim();
      if (input.isEmpty()) { fi0 = 1; fi1 = 2; break; }

      String[] parts = input.split(",");
      if (parts.length != 2) { System.out.println("Enter exactly two indices."); continue; }

      try {
        fi0 = Integer.parseInt(parts[0].trim());
        fi1 = Integer.parseInt(parts[1].trim());
      } catch (NumberFormatException e) { System.out.println("Invalid input."); continue; }

      if (fi0 < 0 || fi0 > max || fi1 < 0 || fi1 > max) { System.out.printf("Indices must be between 0 and %d.%n", max); continue; }
      if (fi0 == fi1) { System.out.println("Indices must be different."); continue; }
      break;
    }

    plotHyperplane(fi0, fi1);
  }

  private void plotAll() {
    if (perceptron == null || perceptronSets == null) {
      System.out.println("Train the perceptron first.");
      return;
    }

    System.out.println("Features: 0 and 1");
    plotHyperplane(0,1);
    System.out.println("Features: 1 and 2");
    plotHyperplane(1,2);
    System.out.println("Features: 2 and 3");
    plotHyperplane(2,3);
    System.out.println("Features: 0 and 2");
    plotHyperplane(0,2);
    System.out.println("Features: 1 and 3");
    plotHyperplane(1,3);
    System.out.println("Features: 0 and 3");
    plotHyperplane(0,3);
  }

  private void plotHyperplane(int fi0, int fi1) {
    int width = 60, height = 20;

    Vector[] filtered = Arrays.stream(dataset)
        .filter(v -> !v.category.equals("virginica"))
        .toArray(Vector[]::new);

    double maxX = Arrays.stream(filtered).mapToDouble(v -> v.data[fi0]).max().getAsDouble();
    double maxY = Arrays.stream(filtered).mapToDouble(v -> v.data[fi1]).max().getAsDouble();

    String RESET         = "\033[0m";
    String BG_DARK       = "\033[100m";  // predict == 0
    String BG_LIGHT      = "\033[47m";   // predict == 1
    String FG_SETOSA     = "\033[92m";   // bright green
    String FG_VERSICOLOR = "\033[91m";   // bright red

    int[][] predictions = new int[height][width];
    char[][] grid = new char[height][width];

    for (int row = 0; row < height; row++) {
      for (int col = 0; col < width; col++) {
        double x = (col / (double)(width - 1)) * maxX;
        double y = ((height - 1 - row) / (double)(height - 1)) * maxY;

        double[] data = new double[perceptron.dimension];
        data[fi0] = x;
        data[fi1] = y;

        predictions[row][col] = perceptron.predict(new Vector(data));
        grid[row][col] = ' ';
      }
    }

    for (Vector v : perceptronSets.test()) {
      int col = (int)((v.data[fi0] / maxX) * (width - 1));
      int row = height - 1 - (int)((v.data[fi1] / maxY) * (height - 1));
      grid[row][col] = v.category.equals("setosa") ? 'S' : 'V';
    }

    System.out.println();
    for (int row = 0; row < height; row++) {
      System.out.print("│");
      for (int col = 0; col < width; col++) {
        String bg = predictions[row][col] == 1 ? BG_LIGHT : BG_DARK;
        char c = grid[row][col];
        if (c == 'S') System.out.print(bg + FG_SETOSA + c + RESET);
        else if (c == 'V') System.out.print(bg + FG_VERSICOLOR + c + RESET);
        else System.out.print(bg + " " + RESET);
      }
      System.out.println();
    }
    System.out.println("└" + "─".repeat(width));
  }
}
