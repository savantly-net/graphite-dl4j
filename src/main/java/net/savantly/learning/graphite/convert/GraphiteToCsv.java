package net.savantly.learning.graphite.convert;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.savantly.learning.graphite.domain.GraphiteDatapoint;
import net.savantly.learning.graphite.domain.GraphiteMultiSeries;
import net.savantly.learning.graphite.domain.GraphiteSeries;

public class GraphiteToCsv {
	private static final Logger log = LoggerFactory.getLogger(GraphiteToCsv.class);

	private File baseDir;

	/**
	 * 
	 * @param workingDir
	 *            Base directory to store the data.
	 */
	public GraphiteToCsv(String workingDir) {
		baseDir = new File(workingDir);
		baseDir.mkdirs();
	}

	public static GraphiteToCsv get(String workingDirectory) {
		return new GraphiteToCsv(workingDirectory);
	}

	public void createFile(String fileName, Pair<String, GraphiteMultiSeries>... series) throws IOException {
		File targetFile = new File(this.baseDir, fileName);
		FileWriter writer = new FileWriter(targetFile);
		Arrays.stream(series).forEach(s -> {
			s.getValue().stream().forEach(g -> {
				g.getDatapoints().stream().forEach(d -> {
					if (d.getValue() != null) {
						String line = String.format("%s,%s,%s,%s\n", d.getEpoc(), d.getValue(), g.getTarget(),
								s.getKey());
						try {
							writer.write(line);
						} catch (IOException e) {
							log.error("failed to write to file: ", e);
						}
					}
				});
			});
		});
		writer.close();
	}
	
	public CsvResult createFileSequence(List<Pair<String, GraphiteMultiSeries>> series) throws IOException {
		return createFileSequence(series, 0.75);
	}

	/**
	 * 
	 * @param series A list of labeled GraphiteMultiSeries data
	 * @return the number of feature/label file pairs
	 * @throws IOException
	 */
	public CsvResult createFileSequence(List<Pair<String, GraphiteMultiSeries>> series, double trainPercentage) throws IOException {

		AtomicInteger filesGeneratedForTraining = new AtomicInteger(0);
		AtomicInteger counter = new AtomicInteger(1);

		Stream<GraphiteSeries> targetGroupsStream = series.stream().flatMap(s -> {
			return s.getValue().stream().map(g -> {
				g.getDatapoints().forEach(d->{
					d.setLabel(s.getKey());
				});
				return g;
			});
		});
		
		Map<String, List<GraphiteSeries>> targetGroups = targetGroupsStream.collect(Collectors.groupingBy(GraphiteSeries::getTarget));
		
		AtomicBoolean createTestData = new AtomicBoolean(false);
		
		AtomicLong countToGenerateForTraining = new AtomicLong(Math.round(targetGroups.size() * trainPercentage));
		if (countToGenerateForTraining.get() == targetGroups.size() && countToGenerateForTraining.get() > 1) {
			countToGenerateForTraining.decrementAndGet();
		} else if (countToGenerateForTraining.get() == targetGroups.size()) {
			throw new RuntimeException("unable to create test data. use a smaller percentage of the data for training");
		}
		targetGroups.forEach((k,g) -> {
			log.trace("counter: {}, trainCount: {}, trainFileCount: {}", counter, countToGenerateForTraining, filesGeneratedForTraining);
			String featuresFileName;
			String labelsFileName;
			if(!createTestData.get()) {
				featuresFileName = String.format("%s.%s.%s", counter.get(), "train", "features.csv");
				labelsFileName = String.format("%s.%s.%s", counter.get(), "train", "labels.csv");
				filesGeneratedForTraining.incrementAndGet();
				if (counter.get() >= countToGenerateForTraining.get() ) {
					createTestData.set(true);
					counter.set(0);
				} else {
					filesGeneratedForTraining.set(counter.get());
				}
			} else {
				featuresFileName = String.format("%s.%s.%s", counter.get(), "test", "features.csv");
				labelsFileName = String.format("%s.%s.%s", counter.get(), "test", "labels.csv");
			}
			File featuresFile = new File(this.baseDir, featuresFileName);
			File labelsFile = new File(this.baseDir, labelsFileName);
			Stream<GraphiteDatapoint> datapointStream = g.stream().flatMap(d -> {
				return d.getDatapoints().stream();
			});
			try {
				// sort by epoch, and strip points where the value is valid
				List<GraphiteDatapoint> sortedDataPoints = datapointStream.sorted().filter(d -> d.getValue() != null && !Double.isNaN(d.getValue().doubleValue())).collect(Collectors.toList());
				AtomicLong dataPointCounter = new AtomicLong(sortedDataPoints.size());
				if (dataPointCounter.get() > 0) {
					counter.incrementAndGet();
					FileWriter labelsWriter = new FileWriter(labelsFile);
					FileWriter featuresWriter = new FileWriter(featuresFile);
					sortedDataPoints.forEach(d -> {
						long leftOver = dataPointCounter.getAndDecrement();
						try {
							labelsWriter.write(String.format("%s", d.getLabel()));
							featuresWriter.write(String.format("%s", d.getValue()));
							if (leftOver > 1) {
								labelsWriter.write("\n");
								featuresWriter.write("\n");
							}
						} catch (IOException e) {
							log.error("{}", e);
							throw new RuntimeException(e);
						}
					});
					labelsWriter.close();
					featuresWriter.close();
				} else {
					// There were no valid datapoints so we delete the file and decrement the counter
					featuresFile.delete();
					labelsFile.delete();
				}

			} catch (IOException e) {
				log.error("{}", e);
				throw new RuntimeException(e);
			}
		});
		
		return new CsvResult(filesGeneratedForTraining.get(), counter.get()-1);
	}
	
	public class CsvResult {
		int trainFileCount;
		int testFileCount;
		public CsvResult(int trainFileCount, int testFileCount) {
			this.trainFileCount = trainFileCount;
			this.testFileCount = testFileCount;
		}
		public int getTrainFileCount() {
			return trainFileCount;
		}
		public int getTestFileCount() {
			return testFileCount;
		}
	}
}
