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

  public static double measureAccuracy(Observation[] testSet, NaiveBayes nb) {
    int correct = 0;
    for (Observation obs : testSet) {
      var prediction = nb.predict(obs);
      if (prediction.equals(obs.label)) correct++;
    }
    return (double) correct / testSet.length * 100;
  }

  public static double measure(Observation[] testSet, NaiveBayes nb, String targetClass, MeasureType mt) {
    int TP = 0, FP = 0, FN = 0;
    for (Observation obs : testSet) {
      String predicted = nb.predict(obs);
      boolean isTarget = obs.label.equals(targetClass);
      boolean predictedTarget = predicted.equals(targetClass);
      if (predictedTarget && isTarget) TP++;
      else if (predictedTarget && !isTarget && mt != MeasureType.Recall) FP++;
      else if (!predictedTarget && isTarget && mt != MeasureType.Precision) FN++;
    }
    double precision = TP + FP == 0 ? 0 : (double) TP / (TP + FP);
    double recall    = TP + FN == 0 ? 0 : (double) TP / (TP + FN);
    return switch (mt) {
      case Precision -> precision;
      case Recall    -> recall;
      case FMeasure  -> getFMeasure(precision, recall);
    };
  }

  public static double measurePrecision(Observation[] testSet, NaiveBayes nb, String targetClass) {
    return measure(testSet, nb, targetClass, MeasureType.Precision);
  }

  public static double measureRecall(Observation[] testSet, NaiveBayes nb, String targetClass) {
    return measure(testSet, nb, targetClass, MeasureType.Recall);
  }

  public static double getFMeasure(Observation[] testSet, NaiveBayes nb, String targetClass) {
    return measure(testSet, nb, targetClass, MeasureType.FMeasure);
  }
}
