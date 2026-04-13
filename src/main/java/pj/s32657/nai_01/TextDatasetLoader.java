package pj.s32657.nai_01;

import java.io.*;
import java.util.*;
import java.nio.file.*;
import java.net.URISyntaxException;
import java.util.stream.*;

record TextDataset(Vector[] vectors, String[] labels) {}

public class TextDatasetLoader {
  private static final int CHUNK_SIZE = 200;
  // exclude Q, W, X due to it not being used by most languages
  private static final char[] LETTERS = "abcdefghijklmnoprstuvyz".toCharArray();

  public static TextDataset load() throws IOException, URISyntaxException {
    var langDir = TextDatasetLoader.class.getResource("/lang");
    assert langDir != null;

    List<Vector> vectors = new ArrayList<>();
    List<String> labels = new ArrayList<>();

    try (var files = Files.list(Paths.get(langDir.toURI()))) {
      for (Path file : files.toList()) {
        String filename = file.getFileName().toString();
        if (!filename.endsWith(".txt")) continue;

        String lang = filename.replace(".txt", "");
        labels.add(lang);
        vectors.addAll(loadLanguage(file, lang));
      }
    }
    return new TextDataset(vectors.toArray(new Vector[0]), labels.toArray(new String[0]));
  }

  private static List<Vector> loadLanguage(Path file, String lang) throws IOException {
    String text;

    // filter out headers
    try (var lines = Files.lines(file)) {
      var filtered = lines.filter(line -> !line.startsWith("#"));
      text = String.join("\n", filtered.toList());
    }

    List<Vector> vectors = new ArrayList<>();
    for (String chunk : chunkify(text))
      vectors.add(vectorize(chunk, lang));

    return vectors;
  }

  private static List<String> chunkify(String text) {
    List<StringBuilder> chunks = new ArrayList<>();
    StringBuilder current = new StringBuilder();

    for (String paragraph : text.split("\n")) {
      for (String sentence : paragraph.trim().split("[.!?]")) {
        current.append(sentence).append(" ");
        if (current.length() < CHUNK_SIZE) continue;

        chunks.add(current);
        current = new StringBuilder();
      }
    }

    if (current.length() >= CHUNK_SIZE/2) chunks.add(current);
    else if (!chunks.isEmpty()) chunks.getLast().append(current);

    return chunks.stream().map(sb -> sb.toString().trim()).toList();
  }
  
  public static Vector vectorize(String chunk, String label) {
    double[] freq = new double[LETTERS.length];
    int total = 0;

    for (char c : chunk.toLowerCase().toCharArray()) {
      for (int i = 0; i < LETTERS.length; i++) {
        if (c == LETTERS[i]) { freq[i]++; total++; break; }
      }
    }

    if (total == 0) return new Vector(freq, label);

    for (int i = 0; i < LETTERS.length; i++) freq[i] /= total;
    return new Vector(freq, label);
  }
}