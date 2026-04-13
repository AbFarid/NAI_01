package pj.s32657.nai_01;

import java.text.DecimalFormat;
import java.util.*;
import java.util.stream.*;

record EpochResults(int errCount, double[] accMeasures, double accAvg) {}
record TrainResult(int epochs, boolean converged, double accAvg) {}

public class Perceptron {
  String label; // target class label
  int dimension;
  double threshold;
  double alpha; // learning rate
  double beta;  // bias

  Vector W;

  public Perceptron(
      int dimension_,
      double weights_,
      double threshold_,
      double alpha_
//    double beta_
  ) {
    dimension = dimension_;
    threshold = threshold_;
    alpha = alpha_;
//  beta = beta_;

    double[] weights = new double[dimension];
    Arrays.fill(weights, weights_);
    W = new Vector(weights);
  }

  public Perceptron(double[] weights_, double threshold_, double alpha_) {
    dimension = weights_.length;
    threshold = threshold_;
    alpha = alpha_;
    W = new Vector(weights_);
  }

  public Perceptron(
      String label_,
      int dimension_,
      double weights_,
      double threshold_,
      double alpha_
  ) {
    this(dimension_, weights_, threshold_, alpha_);
    label = label_;
  }
  public Perceptron(String label_, double[] weights_, double threshold_, double alpha_) {
    this(weights_, threshold_, alpha_);
    label = label_;
  }


  static double[] randomWeights(int dimension) {
    Random rng = new Random();
    double[] weights = new double[dimension];
    for (int i = 0; i < dimension; i++)
      weights[i] = rng.nextDouble() - 0.5;
    return weights;
  }

  int predict(Vector x) { return W.dot(x) >= threshold ? 1 : 0; } // discrete
  double predictCont(Vector x) { return W.dot(x) - threshold; } // continuous

  // discrete
  boolean learn(Vector x) { return learn(x, alpha); }
  boolean learn(Vector x, double alpha) {
    int predicted = predict(x); // activated?
    int expected = x.category.equals(label) ? 1 : 0;

    double delta = (expected - predicted) * alpha;
    boolean error = delta != 0;
    if (error) {
      Vector u = x.scale(delta);
      W = W.add(u);
      threshold = threshold - delta;
    }

    return error;
  }


  EpochResults runEpoch(Vector[] dataset) {
    int oopsCount = 0;
    double[] measures = new double[dataset.length];

    for (int i = 0; i < dataset.length; i++) {
      Vector x = dataset[i];
      boolean error = learn(x);
      if (error) oopsCount++;
      double accuracy = EvaluationMetrics.measureAccuracy(dataset, this);
      measures[i] = accuracy;
    }

    double avg = Arrays.stream(measures).average().getAsDouble();
    return new EpochResults(oopsCount, measures, avg);
  }

  TrainResult train(SplitDataset dataset, String label) { return train(dataset, label, 100, false); }
  TrainResult train(SplitDataset dataset, String label, boolean debug) { return train(dataset, label, 100, debug); }
  TrainResult train(SplitDataset dataset, String label, int maxEpochs) { return train(dataset, label, maxEpochs, false); }
  TrainResult train(SplitDataset dataset, String label, int maxEpochs, boolean debug) {
    this.label = label;
    int epochs = 0;
    EpochResults lastResult = null;

    Vector[] trainingSet = dataset.train();
    Vector[] testSet = dataset.test();
    boolean hadOops = true;

    while (hadOops && epochs < maxEpochs) {
      lastResult = runEpoch(trainingSet);
      epochs += 1;

      if (lastResult.errCount() == 0) hadOops = false;
      if (debug) printReport(epochs, dataset, lastResult);
    }

    return new TrainResult(epochs, !hadOops, lastResult.accAvg());
  }

  void printReport(int epoch, SplitDataset dataset, EpochResults results) {
    DecimalFormat df = new DecimalFormat("0.##");
    Vector[] set = dataset.test();
//    set = dataset.train();

    double accuracy = EvaluationMetrics.measureAccuracy(set, this);
    String wStr = Arrays.stream(W.data).mapToObj(w -> df.format(w)).collect(Collectors.joining(", ", "[", "]"));

    System.out.printf("Epoch %d | Final accuracy: %s%% | Avg. accuracy: %s%% | Errors: %d | W=%s | θ=%s%n",
        epoch, df.format(accuracy), df.format(results.accAvg()), results.errCount(), wStr, df.format(threshold));

//    System.out.print("  Measures: ");
//    Arrays.stream(results.accMeasures()).mapToObj(d -> df.format(d) + " ").forEach(System.out::print);
//    System.out.println();
  }
}