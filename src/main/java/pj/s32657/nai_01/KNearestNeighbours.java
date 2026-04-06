package pj.s32657.nai_01;

import java.util.*;

public class KNearestNeighbours {
  private final Vector[] dataset;
  private int k = 3;

  public KNearestNeighbours(Vector[] vectors, int k_) {
    this.dataset = vectors;
    this.k = k_;
  }

  public KNearestNeighbours(Vector[] dataset) {
    this.dataset = dataset;
  }

  public String categorize(Vector u) {
    if (dataset.length == 0)
      throw new IllegalArgumentException("Dataset is empty");

    if (dataset[0].size != u.size)
      throw new IllegalArgumentException("Dataset size mismatch");

    var distances = new Neighbor[dataset.length];
    for (int i = 0; i < dataset.length; i++) {
      distances[i] = new Neighbor(dataset[i], u.getDist(dataset[i]));
    }

    
    sortByDistance(distances);
    List<String> mode = getMode(distances);

    int random = (int) (Math.random() * mode.size());
    return mode.get(random);
  }

  private void sortByDistance(Neighbor[] neighbors) {
//    Arrays.sort(neighbors, Comparator.comparingDouble(Neighbor::distance));

    for (int i = 1; i < neighbors.length; i++) {
      for (int j = i; j > 0 && neighbors[j - 1].distance() > neighbors[j].distance(); j--) {
        Neighbor temp = neighbors[j];
        neighbors[j] = neighbors[j - 1];
        neighbors[j - 1] = temp;
      }
    }
  }

  private List<String> getMode(Neighbor[] distances) {
    Neighbor[] closest = Arrays.copyOfRange(distances, 0, k);

    Map<String, Integer> occurrences = new HashMap<>();
    for (int i = 0; i < k; i++) {
      String category = closest[i].vector().category;
      occurrences.put(category, occurrences.getOrDefault(category, 0) + 1);
    }

    int max = Collections.max(occurrences.values());
    List<String> mode = new ArrayList<>();
    for (var entry : occurrences.entrySet())
      if (entry.getValue() == max) mode.add(entry.getKey());
    return mode;
  }
}
