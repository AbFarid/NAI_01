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

  public static double measureAccuracy(Vector[] testSet, Perceptron perceptron, Map<String, Integer> labelMap) {
    int correct = 0;
    for (Vector v : testSet) {
      int expected = labelMap.get(v.category);
      int predicted = perceptron.predict(v);
      if (predicted == expected) correct++;
    }

    return (double) correct / testSet.length * 100;
  }

  public static double measurePrecision(Vector[] dataset, Perceptron perceptron, Map<String, Integer> labelMap) {
    // Precision is True Positives / (True Positives + False Positives)

    int TP = 0; // True Positives
    int FP = 0; // False Positives
//  int TN = 0; // True Negatives
//  int FN = 0; // False Negatives

    for (Vector v : dataset) {
      int predicted = perceptron.predict(v);
      int expected =  labelMap.get(v.category);

      if (predicted == 0) continue;
      if (expected == 1) TP++;
      else FP++;
    }

    if (TP + FP == 0) return 0;
    return (double) TP / (TP + FP);
  }

  public static double measureRecall(Vector[] dataset, Perceptron perceptron, Map<String, Integer> labelMap) {
    // Recall is True Positives / (True Positives + False Negatives)

    int TP = 0; // True Positives
    int FN = 0; // False Negatives

    for (Vector v : dataset) {
      int predicted = perceptron.predict(v);
      int expected = labelMap.get(v.category);

      if (expected == 0) continue;
      if (predicted == 1) TP++;
      else FN++;
    }

    if (TP + FN == 0) return 0;
    return (double) TP / (TP + FN);
  }

  public static double getFMeasure(double precision, double recall) {
    if (precision + recall == 0) return 0;
    return (2 * precision * recall) / (precision + recall);
  }

  public static double getFMeasure(Vector[] dataset, Perceptron perceptron, Map<String, Integer> labelMap) {
    double precision = measurePrecision(dataset, perceptron, labelMap);
    double recall = measureRecall(dataset, perceptron, labelMap);
    return getFMeasure(precision, recall);
  }
}
