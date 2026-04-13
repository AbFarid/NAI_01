package pj.s32657.nai_01;

import java.util.*;

public class EvaluationMetrics {

  public static double measureAccuracy(SplitDataset subsets) {
    return measureAccuracy(subsets, 3);
  }

  public static double measureAccuracy(SplitDataset subsets, int k) {
    KNearestNeighbours knn = new KNearestNeighbours(subsets.train(), k);

    int correct = 0;
    for (Vector v : subsets.test()) {
      if (knn.categorize(v).equals(v.category))
        correct++;
    }

    return (double) correct / subsets.test().length * 100;
  }

  public static double measureAccuracy(Vector[] testSet, Perceptron perceptron) {
    int correct = 0;
    for (Vector v : testSet) {
      int expected = v.category.equals(perceptron.label) ? 1 : 0;
      int predicted = perceptron.predict(v);
      if (predicted == expected) correct++;
    }

    return (double) correct / testSet.length * 100;
  }

  enum MeasureType { Precision, Recall, FMeasure }

  public static double measure(Vector[] dataset, Perceptron perceptron, MeasureType mt) {
    // Precision is True Positives / (True Positives + False Positives)

    int TP = 0; // True Positives
    int FP = 0; // False Positives
    int FN = 0; // False Negatives

    double precision = 0;
    double recall = 0;

    for (Vector v : dataset) {
      int predicted = perceptron.predict(v);
      int expected = v.category.equals(perceptron.label) ? 1 : 0;

      if (predicted == 1 && expected == 1) { TP++; continue; }
      if (predicted == 1 && expected == 0 && mt != MeasureType.Recall) FP++;
      if (predicted == 0 && expected == 1 && mt != MeasureType.Precision) FN++;
    }

    if (mt != MeasureType.Recall) precision = TP + FP == 0 ? 0 : (double) TP / (TP + FP);
    if (mt != MeasureType.Precision) recall = TP + FN == 0 ? 0 : (double) TP / (TP + FN);

    return switch (mt) {
      case Precision  -> precision;
      case Recall     -> recall;
      case FMeasure   -> precision + recall == 0 ? 0 : (2 * precision * recall) / (precision + recall);
    };
  }

  public static double measurePrecision(Vector[] dataset, Perceptron perceptron) {
    // Precision is True Positives / (True Positives + False Positives)
    return measure(dataset, perceptron, MeasureType.Precision);
  }

  public static double measureRecall(Vector[] dataset, Perceptron perceptron) {
    // Recall is True Positives / (True Positives + False Negatives)
    return measure(dataset, perceptron, MeasureType.Recall);
  }

  public static double getFMeasure(Vector[] dataset, Perceptron perceptron) {
    return measure(dataset, perceptron, MeasureType.FMeasure);
  }
  public static double getFMeasure(double precision, double recall) {
    if (precision + recall == 0) return 0;
    return (2 * precision * recall) / (precision + recall);
  }
}
