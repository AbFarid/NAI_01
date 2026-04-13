package pj.s32657.nai_01;

import java.util.*;
import java.util.concurrent.*;

record Prediction(String winner, Map<String, Double> scores) {}
record LayerTrainResult(List<TrainResult> results, String[] labels, long timeMs) {}

public class SingleLayerNN {
  Perceptron[] neurons;
  double alpha;
  double beta;

  public SingleLayerNN(Perceptron[] neurons, double alpha, double beta) {
    this.neurons = neurons;
    this.alpha = alpha;
    this.beta = beta;
  }

  public LayerTrainResult train(SplitDataset dataset, String[] labels) { return train(dataset, labels, 100, false); }
  public LayerTrainResult train(SplitDataset dataset, String[] labels, int maxEpochs) { return train(dataset, labels, maxEpochs, false); }
  public LayerTrainResult train(SplitDataset dataset, String[] labels, boolean debug) { return train(dataset, labels, 100, debug); }
  public LayerTrainResult train(SplitDataset dataset, String[] labels, int maxEpochs, boolean debug) {
    if (neurons.length != labels.length)
      throw new IllegalArgumentException("Neurons count not equal to labels count.");

    List<TrainResult> results = new ArrayList<>();
    long start = System.currentTimeMillis();

    if (debug) {
      for (int i = 0; i < labels.length; i++) {
        System.out.println("\n-- Training: " + labels[i].toUpperCase() + " --");
        results.add(neurons[i].train(dataset, labels[i], maxEpochs, true));
        System.out.println("-- Done: " + labels[i].toUpperCase() + " --");
      }
      return new LayerTrainResult(results, labels, System.currentTimeMillis() - start);
    }

    try (ExecutorService pool = Executors.newFixedThreadPool(neurons.length)) {
      List<Future<TrainResult>> futures = new ArrayList<>();

      int currentNeuron = 0;
      for (String label : labels) {
        final int idx = currentNeuron++;
        futures.add(pool.submit(() -> neurons[idx].train(dataset, label, maxEpochs, false)));
      }

      pool.shutdown();
      pool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);

      for (Future<TrainResult> f : futures) results.add(f.get());
    }
    catch (InterruptedException | ExecutionException e) { e.printStackTrace(); }

    return new LayerTrainResult(results, labels, System.currentTimeMillis() - start);
  }

  public Prediction predict(Vector x) {
    Map<String, Double> scores = new HashMap<>();

    for (Perceptron n : neurons)
      scores.put(n.label, n.predictCont(x));

    String winner = Collections
        .max(scores.entrySet(), Map.Entry.comparingByValue())
        .getKey();

    return new Prediction(winner, scores);
  }
}