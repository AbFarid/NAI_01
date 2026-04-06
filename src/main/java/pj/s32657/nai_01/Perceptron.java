package pj.s32657.nai_01;

import java.text.DecimalFormat;
import java.util.*;
import java.util.function.*;
import java.util.stream.*;

record EpochResults(int errCount, double[] accMeasures, double accAvg) {}

public class Perceptron {
  int dimension;
  double[] weights;
  double threshold;
  double alpha; // learning rate
  double beta;  // bias

  Vector W;

  public Perceptron(
      int dimension_,
      double weights_,
      double threshold_,
      double alpha_
//      double beta_
  ) {
    dimension = dimension_;
    threshold = threshold_;
    alpha = alpha_;
//    beta = beta_;

    weights = new double[dimension];
    Arrays.fill(weights, weights_);
    W = new Vector(weights);
  }

  public Perceptron(
      double[] weights_,
      double threshold_,
      double alpha_
  ) {
    weights = weights_;
    dimension = weights_.length;
    threshold = threshold_;
    alpha = alpha_;

    W = new Vector(weights);
  }

  public Perceptron(Vector weights_, double threshold_, double alpha_) {
    W = weights_;
    dimension = weights_.size;
    threshold = threshold_;
    alpha = alpha_;
  }

  int predict(Vector x) { // discrete
    return W.dot(x) >= threshold ? 1 : 0;
  }

  double predict_cont(Vector x) { // continuous
    return W.dot(x) - threshold;
  }

  boolean learn(Vector x, int expected, double alpha) { // discrete
    int predicted = predict(x); // activated?

    double delta = (expected - predicted) * alpha;
    boolean error = delta != 0;
    if (error) {
      Vector u = x.scale(delta);
      W = W.add(u);
      threshold = threshold - delta;
    }

    return error;
  }

  boolean learn(Vector x, int expected) { // discrete
    return learn(x, expected, alpha);
  }

  EpochResults runEpoch(
      Vector[] dataset,
      Map<String, Integer> mapper // ex.: Map.of("setosa", 1, "versicolor", 0)
  ) {
    int oopsCount = 0;
    double[] measures = new double[dataset.length];

    for (int i = 0; i < dataset.length; i++) {
      Vector x = dataset[i];
      boolean error = learn(x, mapper.get(x.category));
      if (error) oopsCount++;
      double accuracy = EvaluationMetrics.measureAccuracy(dataset, this, mapper);
      measures[i] = accuracy;
    }

    double avg = Arrays.stream(measures).average().getAsDouble();
    return new EpochResults(oopsCount, measures, avg);
  }

  void train(SplitDataset dataset, Map<String, Integer> mapper) { train(dataset, mapper, 100, false); }
  void train(SplitDataset dataset, Map<String, Integer> mapper, int maxEpochs, boolean debug) {
    int epochs = 0;

    Vector[] trainingSet = dataset.train();
    Vector[] testSet = dataset.test();
    boolean hadOops = true;

    while (hadOops && epochs < maxEpochs) {
      var results = runEpoch(trainingSet, mapper);
      epochs += 1;

      if (results.errCount() == 0) hadOops = false;
      if (debug) printReport(epochs, dataset, mapper, results);
    }
  }

  void printReport(int epoch, SplitDataset dataset, Map<String, Integer> mapper, EpochResults results) {
    DecimalFormat df = new DecimalFormat("0.##");
    Vector[] set = dataset.test();
//    set = dataset.train();

    double accuracy = EvaluationMetrics.measureAccuracy(set, this, mapper);
    String wStr = Arrays.stream(W.data).mapToObj(w -> df.format(w)).collect(Collectors.joining(", ", "[", "]"));

    System.out.printf("Epoch %d | Final accuracy: %s%% | Avg. accuracy: %s%% | Errors: %d | W=%s | θ=%s%n",
        epoch, df.format(accuracy), df.format(results.accAvg()), results.errCount(), wStr, df.format(threshold));

    System.out.print("  Measures: ");
    Arrays.stream(results.accMeasures()).mapToObj(d -> df.format(d) + " ").forEach(System.out::print);
    System.out.println();
  }
}
