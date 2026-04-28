package pj.s32657.nai_01;

import java.util.*;

public class UserInterface {
  private Vector[] dataset;
  private KNearestNeighbours knn;

  private Perceptron perceptron;
  private SplitDataset perceptronSets;

  private SingleLayerNN slnn;
  private SplitDataset slnnSets;

  private NaiveBayes nb;
  private SplitObsDataset nbDataset;

  private final Scanner scanner = new Scanner(System.in);

  String RESET    = "\033[0m";
  String BG_DARK  = "\033[100m";
  String BG_LIGHT = "\033[47m";
  String GREEN    = "\033[92m";
  String ORANGE   = "\033[93m";
  String RED      = "\033[91m";

  public UserInterface() {
    DatasetLoader loader = new DatasetLoader();

    try { dataset = loader.load(); }
    catch (Exception e) { e.printStackTrace(); }
  }

  public void run() {
    while (true) {
      System.out.println("\n1. KNN");
      System.out.println("2. Perceptron");
      System.out.println("3. Single-layer Neural Network");
      System.out.println("4. Naive Bayes");
      System.out.println("0. Exit");
      System.out.print("> ");

      switch (scanner.nextLine().trim()) {
        case "1" -> knn();
        case "2" -> perceptron();
        case "3" -> SLNN();
        case "4" -> naiveBayes();
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
//    var mapper = Map.of("versicolor", 0, "setosa", 1);
    perceptron.train(perceptronSets, "setosa", 100, true);
  }

  private void predictPerceptron() {
    if (perceptron == null || perceptronSets == null) {
      System.out.println("Train the perceptron first.");
      return;
    }

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
        if (c == 'S') System.out.print(bg + GREEN + c + RESET);
        else if (c == 'V') System.out.print(bg + RED + c + RESET);
        else System.out.print(bg + " " + RESET);
      }
      System.out.println();
    }
    System.out.println("└" + "─".repeat(width));
  }

  private void SLNN() {
    while (true) {
      System.out.println("\n-- Single-Layer Neural Network --");
      System.out.println("1. Train");
      System.out.println("2. Predict");
      System.out.println("3. Metrics");
      System.out.println("0. Back");
      System.out.print("> ");

      switch (scanner.nextLine().trim()) {
        case "1" -> trainSLNN();
        case "2" -> predictSLNN();
        case "3" -> metricsSLNN();
        case "0" -> { return; }
        default  -> System.out.println("Invalid option.");
      }
    }
  }

  private void trainSLNN() {
    System.out.print("Epochs, debug [default: 1000, false]: ");
    String input = scanner.nextLine().trim();

    int epochs = 1000;
    boolean debug = false;

    if (!input.isEmpty()) {
      String[] parts = input.split(",");
      if (!parts[0].trim().isEmpty())
        epochs = Integer.parseInt(parts[0].trim());
      if (parts.length > 1)
        debug = parts[1].trim().matches("t|true");
    }

    TextDataset textData;
    try { textData = TextDatasetLoader.load(); }
    catch (Exception e) { System.err.println("Failed to load data: " + e.getMessage()); return; }

    slnnSets = new PrepareDataset().trainTestSplit(textData.vectors(), 0.7);
    String[] labels = textData.labels();

    Perceptron[] neurons = new Perceptron[labels.length];
    for (int i = 0; i < labels.length; i++)
      neurons[i] = new Perceptron(labels[i], Perceptron.randomWeights(23), 0.25, 0.01);

    slnn = new SingleLayerNN(neurons, 0.01, 0);
    LayerTrainResult trainResult = slnn.train(slnnSets, labels, epochs, debug);
    printTrainSummary(trainResult);
  }

  private void printTrainSummary(LayerTrainResult result) {
    final int WIDTH = 38;
    System.out.println();
    System.out.printf("%-5s %8s %8s %14s%n", "LANG", "Acc", "Epochs", "Stop");
    System.out.println("-".repeat(WIDTH));

    for (int i = 0; i < result.labels().length; i++) {
      TrainResult r = result.results().get(i);
      String accColor = r.accAvg() >= 100 ? GREEN : r.accAvg() >= 90 ? ORANGE : RED;
      String stop = r.converged() ? "Converged" : "Epoch limit";
      System.out.printf("%-5s %s%7.2f%%%s %8d %14s%n",
          result.labels()[i].toUpperCase(), accColor, r.accAvg(), RESET, r.epochs(), stop);
    }

    System.out.println("-".repeat(WIDTH));
    System.out.printf("Total training time: %s%.3f seconds%s%n", ORANGE, result.timeMs() / 1000.0, RESET);
  }

  private void predictSLNN() {
    if (slnn == null) { System.out.println("Train the network first."); return; }

    while (true) {
      System.out.print("Enter text (0 to stop): ");
      String input = scanner.nextLine().trim();
      if (input.equals("0")) return;

      Vector v = TextDatasetLoader.vectorize(input, "");
      Prediction result = slnn.predict(v);

      System.out.println("Detected language: " + GREEN + result.winner().toUpperCase() + RESET);
      System.out.println();

      double min = Collections.min(result.scores().values());
      double max = Collections.max(result.scores().values());
      double range = max - min;
      int barWidth = 20;

      for (var entry : result.scores().entrySet()) {
        String label = entry.getKey().toUpperCase();
        double normalized = range == 0 ? 0 : (entry.getValue() - min) / range;
        int filled = (int) (normalized * barWidth);
        String bar = "█".repeat(filled) + "░".repeat(barWidth - filled);

        String color;
        if (entry.getKey().equals(result.winner())) color = GREEN;
        else if (normalized >= 0.5) color = ORANGE;
        else color = RED;

        System.out.printf("%s %s%s%s %.4f%n", label, color, bar, RESET, entry.getValue());
      }

      System.out.println();
    }
  }

  private void metricsSLNN() {
    if (slnn == null) { System.out.println("Train the network first."); return; }

    Vector[] testSet = slnnSets.test();
    final int WIDTH = 41;

    System.out.println();
    System.out.printf("%-5s %8s %8s %8s %8s%n", "LANG", "Acc", "Prec", "Recall", "F-M");
    System.out.println("-".repeat(WIDTH));

    for (Perceptron neuron : slnn.neurons) {
      double acc  = EvaluationMetrics.measureAccuracy(testSet, neuron);
      double prec = EvaluationMetrics.measurePrecision(testSet, neuron);
      double rec  = EvaluationMetrics.measureRecall(testSet, neuron);
      double f1   = EvaluationMetrics.getFMeasure(prec, rec);

      String accColor  = acc  >= 100 ? GREEN : acc  >= 90   ? ORANGE : RED;
      String precColor = prec >= 1.0 ? GREEN : prec >= 0.9  ? ORANGE : RED;
      String recColor  = rec  >= 1.0 ? GREEN : rec  >= 0.9  ? ORANGE : RED;
      String f1Color   = f1   >= 1.0 ? GREEN : f1   >= 0.9  ? ORANGE : RED;

      System.out.printf("%-5s %s%7.2f%%%s %s%8.4f%s %s%8.4f%s %s%8.4f%s%n",
          neuron.label.toUpperCase(),
          accColor, acc, RESET,
          precColor, prec, RESET,
          recColor, rec, RESET,
          f1Color, f1, RESET
      );
    }

    System.out.println("-".repeat(WIDTH));
  }

  private void naiveBayes() {
    while (true) {
      System.out.println("\n-- Naive Bayes --");
      System.out.println("1. Load & train");
      System.out.println("2. Predict");
      System.out.println("3. Metrics");
      System.out.println("4. Toggle smoothing");
      System.out.println("0. Back");
      System.out.print("> ");

      switch (scanner.nextLine().trim()) {
        case "1" -> trainNB();
        case "2" -> predictNB();
        case "3" -> metricsNB();
        case "4" -> setSmoothing();
        case "0" -> { return; }
        default  -> System.out.println("Invalid option.");
      }
    }
  }

  private void setSmoothing() {
    if (nb == null) { System.out.println("Load and train first."); return; }
    boolean enabled = nb.getSmoothingEnabled();
    System.out.print("Apply smoothing to all? [" + (enabled ? "y/N" : "Y/n") + "]: ");
    String input = scanner.nextLine().trim();
    nb.setSmoothingEnabled(input.isEmpty() ? !enabled : input.equalsIgnoreCase("y"));
    System.out.println("Smoothing all: " + (nb.getSmoothingEnabled() ? "ON" : "OFF"));
  }

  private void trainNB() {
    System.out.print("Path to CSV [default: outGame.csv]: ");
    String input = scanner.nextLine().trim();
    String path = input.isEmpty() ? "src/main/resources/outGame.csv" : input;

    Observation[] all;
    try {
      all = new DatasetLoader(path).loadObservations();
    } catch (Exception e) {
      System.err.println("Error: " + e.getMessage());
      return;
    }

    int testCount = promptInt("Rows to reserve for testing [default 2]: ", 2);
    System.out.print("Apply smoothing to all? [y/N]: ");
    boolean smoothAll = scanner.nextLine().trim().equalsIgnoreCase("y");

    nbDataset = new PrepareDataset().trainTestSplit(all, testCount);

    nb = new NaiveBayes(nbDataset.train(), smoothAll);
    System.out.printf(
        "Trained on %d rows, %d reserved for testing.%n",
        nbDataset.train().length,
        nbDataset.test().length
    );
  }

  private void predictNB() {
    if (nb == null) { System.out.println("Load and train first."); return; }

    System.out.print("Enter attribute values separated by commas (outlook, temperature, humidity, windy): ");
    String[] parts = scanner.nextLine().trim().split(",");

    String smoothed = nb.getSmoothingEnabled() ? "ENABLED" : "DISABLED";

    System.out.println("Predicting (smoothing all " + smoothed + "): " + Arrays.toString(parts));

    var attributes = Arrays.stream(parts).map(String::trim).toArray(String[]::new);
    Observation obs = new Observation(attributes);

    System.out.println("Predicted: " + GREEN + nb.predict(obs).toUpperCase() + RESET);
  }

  private void metricsNB() {
    if (nb == null) {
      System.out.println("Load and train first.");
      return;
    }

    final int WIDTH = 42;
    System.out.println();
    System.out.printf("%-6s %8s %8s %8s %8s%n", "CLASS", "Acc", "Prec", "Recall", "F-M");
    System.out.println("-".repeat(WIDTH));

    for (String label : nb.priors.keySet()) {
      double acc = EvaluationMetrics.measureAccuracy(nbDataset.test(), nb);
      double prec = EvaluationMetrics.measurePrecision(nbDataset.test(), nb, label);
      double rec = EvaluationMetrics.measureRecall(nbDataset.test(), nb, label);
      double f1 = EvaluationMetrics.getFMeasure(nbDataset.test(), nb, label);

      System.out.printf("%-6s %7.2f%% %8.4f %8.4f %8.4f%n",
          label.toUpperCase(), acc, prec, rec, f1);
    }

    System.out.println("-".repeat(WIDTH));
  }
}

