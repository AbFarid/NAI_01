package pj.s32657.nai_01;

import java.util.*;
import java.util.stream.*;

class Observation {
  String[] attributes;
  String label = "";

  Observation(String[] attributes) {
    this.attributes = attributes;
  }

  Observation(String[] attributes, String label) {
    this.attributes = attributes;
    this.label = label;
  }
}

class Fraction {
  long numerator;
  long denominator;

  public Fraction(long numerator, long denominator) {
    this.numerator = numerator;
    this.denominator = denominator;
  }

  Fraction smoothed(int uniqueVals) {
    return new Fraction(numerator + 1, denominator + uniqueVals);
  }

  double getDouble() {
    return (double)numerator / denominator;
  }
}

public class NaiveBayes {
  private final Observation[] dataset;
  private boolean applySmoothingAll = false;

  Map<String, Double> priors;
  Map<String, Map<Integer, Map<String, Fraction>>> conditionals;
  Map<Integer, Integer> uniqueValueCounts;

  public NaiveBayes(Observation[] _dataset) { this(_dataset, false);}
  public NaiveBayes(Observation[] _dataset, boolean smoothAll) {
    dataset = _dataset;
    applySmoothingAll = smoothAll;

    computeAPriori();
    computeAPosteriori();
  }

  public boolean getSmoothingEnabled() { return applySmoothingAll; }
  public void setSmoothingEnabled(boolean enabled) { applySmoothingAll = enabled; }

  // UNUSED (implemented directly into Fraction)
  double simpleSmoothing(int numerator, int denominator, int uniqueVals) {
    return (double)(numerator + 1) / (denominator + uniqueVals);
  }

  double measureRSS(double[] labels) {
    if (labels.length == 0) return 0.0;

    double sum = 0.0;
    for (double label : labels) sum += label;

    double avg = sum / dataset.length;

    double output = 0.0;
    for (double label : labels) {
      double pred = label - avg;
      output += pred*pred;
    }

    return output;
  }

  private double P(String attrib, int attribIndex, String label) {
    Fraction f = conditionals.get(label).get(attribIndex).get(attrib);

    if (f.numerator == 0 || applySmoothingAll)
      f = f.smoothed(uniqueValueCounts.get(attribIndex));

    return f.getDouble();
  }

  public String predict(Observation obs) {
    return priors.keySet().stream().max(Comparator.comparingDouble(
        label -> priors.get(label) * IntStream.range(0, obs.attributes.length)
            .mapToDouble(i -> P(obs.attributes[i], i, label))
            .reduce(1.0, (a, b) -> a * b)
    )).orElseThrow();
  }

  private double computeAPriori(String label) {
    if (dataset == null || dataset.length == 0)
      throw new IllegalArgumentException("Dataset is empty");

//    Set<String> uniqueLabels = Arrays.stream(dataset)
//        .map(d -> d.label).collect(Collectors.toSet());
//    labels = new HashMap<>();
//    for (String label : uniqueLabels) labels.put(label, computeAPriori(label));

    int total = dataset.length;
    double count = Arrays.stream(dataset)
        .filter(d -> d.label.equals(label)).count();

    return count / total;
  }

  private void computeAPriori() {
    if (dataset == null || dataset.length == 0)
      throw new IllegalArgumentException("Dataset is empty");

    priors = Arrays.stream(dataset)
        .collect(Collectors.groupingBy(
            d -> d.label,
            Collectors.collectingAndThen(
                Collectors.counting(),
                count -> (double)count / dataset.length
            )
        ));
  }

  private void computeAPosteriori() {
    if (dataset == null || dataset.length == 0)
      throw new IllegalArgumentException("Dataset is empty");

    conditionals = new HashMap<>();
    Map<String, List<Observation>> byLabel = Arrays.stream(dataset)
        .collect(Collectors.groupingBy(d -> d.label));

    int numAttribs = dataset[0].attributes.length;
    List<Set<String>> allValues = new ArrayList<>();
    for (int i = 0; i < numAttribs; i++) {
      final int n = i;
      allValues.add(Arrays.stream(dataset)
          .map(d -> d.attributes[n])
          .collect(Collectors.toSet()));
    }
    
    for (var entry : byLabel.entrySet()) {
      String label = entry.getKey();
      List<Observation> observations = entry.getValue();
      long denominator = observations.size();

      Map<Integer, Map<String, Fraction>> attrMap = new HashMap<>();

      for (int i = 0; i < numAttribs; i++) {
        final int n = i;
        Map<String, Long> valueCounts = observations.stream().collect(
            Collectors.groupingBy(d -> d.attributes[n], Collectors.counting())
        );

        Map<String, Fraction> valueMap = new HashMap<>();
        for (String value : allValues.get(i))
          valueMap.put(value, new Fraction(valueCounts.getOrDefault(value, 0L), denominator));

        attrMap.put(i, valueMap);
      }

      conditionals.put(label, attrMap);
    }

    uniqueValueCounts = new HashMap<>();
    for (int i = 0; i < numAttribs; i++)
      uniqueValueCounts.put(i, allValues.get(i).size());

//        var attribs = observation.attributes[i];
//        final int n = i;
//        var filtered = Arrays.stream(dataset).filter(d -> d.label.equals(observation.label)).toList();
//        long denominator = filtered.size();
//        long numerator = filtered.stream().filter(d -> d.attributes[n].equals(attrib)).count();
//        Fraction probability = new Fraction(numerator, denominator);
//        conditionals.get(observation.label).get(n).put(attrib, probability);
  }

}
