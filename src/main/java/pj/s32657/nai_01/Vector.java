package pj.s32657.nai_01;

public class Vector {
  double[] data;
  int size;
  String category;
  String cluster;

  Vector(double[] u) {
    this.data = u;
    this.size = u.length;
  }

  Vector(double[] u, String category) {
    this.data = u;
    this.size = u.length;
    this.category = category;
  }

  Vector add(Vector vec) {
    if (size != vec.size)
      throw new IllegalArgumentException("Vectors must have same size");

    double[] v = vec.data;
    double[] w = new double[data.length];
    for (int i = 0; i < size; i++)
      w[i] = data[i] + v[i];

    return new Vector(w);
  }

  Vector scale(double scale) {
    double[] v = new double[size];
    for (int i = 0; i < size; i++)
      v[i] = data[i] *  scale;

    return new Vector(v);
  }

  double getNorm() { // Euclidean
    double sum = 0;
    for (int i = 0; i < size; i++)
      sum += data[i] * data[i];

    return Math.sqrt(sum);
  }

  double getDist(Vector v) { //Euclidean
    if (size != v.size)
      throw new IllegalArgumentException("Vectors must have same size");

    double sum = 0;
    for (int i = 0; i < size; i++) {
      double d = data[i] - v.data[i];
      sum += d * d;
    }

    return Math.sqrt(sum);
  }

  double dot(Vector v) {
    if (size != v.size)
      throw new IllegalArgumentException("Vectors must have same size");

    double sum = 0;
    for (int i = 0; i < size; i++)
      sum += data[i] * v.data[i];

    return sum;
  }

  /** EXTRA **/

  static Vector addVectors(Vector u, Vector v) {
    if (u.size != v.size)
      throw new IllegalArgumentException("Vectors must have same size");

    double[] w = new double[u.size];
    for (int i = 0; i < v.size; i++)
      w[i] = u.data[i] + v.data[i];

    return new Vector(w);
  }

  static double dotProdVectors(Vector u, Vector v) {
    if (u.size != v.size)
      throw new IllegalArgumentException("Vectors must have same size");

    double sum = 0;
    for (int i = 0; i < u.size; i++)
      sum += u.data[i] * v.data[i];

    return sum;
  }
}