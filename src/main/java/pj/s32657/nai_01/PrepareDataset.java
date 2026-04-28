package pj.s32657.nai_01;

import java.util.*;
import java.util.function.Predicate;

public class PrepareDataset {

  public SplitDataset trainTestSplit(Vector[] dataset) {
    return trainTestSplit(dataset, 2.0/3.0, null, false);
  }

  public SplitDataset trainTestSplit(Vector[] dataset, double trainRatio) {
    return trainTestSplit(dataset, trainRatio, null, false);
  }

  public SplitDataset trainTestSplit(
      Vector[] dataset,
      double trainRatio,
      Predicate<Vector> filter,
      boolean shuffle
  ) {
    if (filter != null)
      dataset = Arrays.stream(dataset).filter(filter).toArray(Vector[]::new);

    HashMap<String, List<Vector>> categories = new HashMap<>();

    for (Vector vector : dataset) categories
      .computeIfAbsent(vector.category, k -> new ArrayList<>())
      .add(vector);

    List<Vector> trainSet = new ArrayList<>();
    List<Vector> testSet = new ArrayList<>();

    for (var entry : categories.entrySet()) {
      List<Vector> group = entry.getValue();
      Collections.shuffle(group);

      int trainSize = (int) Math.ceil(group.size() * trainRatio);
      trainSet.addAll(group.subList(0, trainSize));
      testSet.addAll(group.subList(trainSize, group.size()));
    }

    if (shuffle) {
      Collections.shuffle(trainSet);
      Collections.shuffle(testSet);
    }

    return new SplitDataset(
      trainSet.toArray(new Vector[0]),
      testSet.toArray(new Vector[0])
    );
  }

  public SplitObsDataset trainTestSplit(Observation[] dataset, int testCount) {
    List<Observation> list = new ArrayList<>(Arrays.asList(dataset));
    Collections.shuffle(list);
    Observation[] test  = list.subList(0, testCount).toArray(new Observation[0]);
    Observation[] train = list.subList(testCount, list.size()).toArray(new Observation[0]);
    return new SplitObsDataset(train, test);
  }
}
