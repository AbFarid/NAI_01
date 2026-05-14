package pj.s32657.nai_01;

import java.util.Iterator;

class Range implements Iterable<Integer> {
  private final int start;
  private final int end;

  Range(int end) { this(0, end); }

  Range(int start, int end) {
    this.start = start;
    this.end = end;
  }

  public Iterator<Integer> iterator() {
    return new Iterator<>() {
      int i = start;
      public boolean hasNext() { return i < end; }
      public Integer next() { return i++; }
    };
  }
}