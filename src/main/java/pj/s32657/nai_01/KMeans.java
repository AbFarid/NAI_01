package pj.s32657.nai_01;

import java.awt.*;
import java.util.*;
import java.util.stream.IntStream;

class Cluster {
  Vector centroid;
  String label;
  ArrayList<Vector> members;
  Color color;


  static Color[] COLORS = {
      new Color(255, 94, 66),   // red
      new Color(131, 213, 0),   // green
      new Color(55, 182, 255),  // blue
      new Color(255, 194, 40),  // orange
      new Color(248, 119, 211), // pink
      new Color(55, 255, 208),  // teal
  };

  static Color colorFor(int i) {
    if (i < COLORS.length) return COLORS[i];
    return new Color((int)(Math.random() * 0x1000000));
  }

  public int size() { return members.getFirst().size; }

  Cluster(String label, Color color) {
    this.label = label;
    this.color = color;
    this.members = new ArrayList<>();
  }

  Cluster(Vector centroid, String label, Color color) {
    this(label, color);
    this.centroid = centroid;
  }

  public Vector calcCentroid() {
    if (members.isEmpty()) return centroid;

    var values = new double[size()];
    for (int i : new Range(0, values.length)) {
      for (Vector v : members) values[i] += v.data[i];
      values[i] /= members.size();
    }

    return centroid = new Vector(values);
  }

}

public class KMeans {
  int k;
  Vector[] points;
  Cluster[] clusters;

  static final int MAX_RUNS = 1000;

  public KMeans(int k, Vector[] points) {
    this.k = k;
    this.points = points;
    randomClustering();
  }

  public KMeans(int k, Vector[] points, Cluster[] clusters) {
    this.k = k;
    this.points = points;
    this.clusters = clusters;
  }

  private void randomClustering() {
    if (points == null || points.length == 0)
      throw new IllegalArgumentException("Points array is null or empty");

    if (points.length < k)
      throw new IllegalArgumentException("k is bigger than the amount of points");

    var random = new Random();

    this.clusters = IntStream.range(0, k).mapToObj(i -> new Cluster(
        String.valueOf(i),
        Cluster.colorFor(i)
    )).toArray(Cluster[]::new);

    ArrayList<Vector> shuffled = new ArrayList<>(Arrays.asList(points));
    Collections.shuffle(shuffled, random);

    for (int i : new Range(k)) {
      shuffled.get(i).cluster = String.valueOf(i);
      clusters[i].members.add(shuffled.get(i));
    }

    for (int i : new Range(k, shuffled.size())) {
      int c = random.nextInt(k);
      shuffled.get(i).cluster = String.valueOf(c);
      clusters[c].members.add(shuffled.get(i));
    }

    for (Cluster c : clusters) c.calcCentroid();
  }

  public Cluster findClosestCluster(Vector v) {
    if (clusters == null || clusters.length == 0)
      throw new RuntimeException("No clusters specified");

    ArrayList<Cluster> candidates = new ArrayList<>();
    double minDist = Double.POSITIVE_INFINITY;
    for (Cluster c : clusters) {
      double dist = v.getDist(c.centroid);

      if (dist < minDist) {
        minDist = dist;
        candidates.clear();
        candidates.add(c);
      }

      else if (dist == minDist) candidates.add(c);
    }

    return candidates.get((new Random().nextInt(0, candidates.size())));
  }

  // this could be optimized for larger k values, but who cares...
  private Cluster clusterFor(String label) {
    return Arrays.stream(clusters).filter(c -> c.label.equals(label)).findFirst().orElseThrow();
  }

  public void train() {
    if (points == null || points.length == 0)
      throw new IllegalArgumentException("Points array is null or empty");

    if (clusters == null || clusters.length == 0) randomClustering();

    int runs = 0;
    boolean migrated = true;
    while (runs < MAX_RUNS && migrated) {
      runs++; migrated = false;

      for (Vector v : points) {
        var closest = findClosestCluster(v);
        if (!closest.label.equals(v.cluster)) {
          migrated = true;
          var current = clusterFor(v.cluster);
          v.cluster = closest.label;

          current.members.remove(v);
          closest.members.add(v);
        }
      }

      for (Cluster c : clusters) c.calcCentroid();
    }

    if (runs >= MAX_RUNS) System.out.println("Stop reason: max iterations reached.");
    else System.out.println("Stop reason: all clusters settled.");
  }

}
