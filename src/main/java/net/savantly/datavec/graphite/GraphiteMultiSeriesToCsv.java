package net.savantly.datavec.graphite;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class GraphiteMultiSeriesToCsv {
	private static final Logger log = LoggerFactory.getLogger(GraphiteMultiSeriesToCsv.class);

	private File baseDir;

	/**
	 * 
	 * @param workingDir
	 *            Base directory to store the data.
	 * @param graphiteMultiSeries
	 *            A list of graphite series to load
	 */
	public GraphiteMultiSeriesToCsv(String workingDir) {
		baseDir = new File(workingDir);
		baseDir.mkdirs();
	}

	public static GraphiteMultiSeriesToCsv get(String workingDirectory) {
		return new GraphiteMultiSeriesToCsv(workingDirectory);
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

	/**
	 * 
	 * @param series
	 * @return the number of feature/label file pairs
	 * @throws IOException
	 */
	public int createFileSequence(List<Pair<String, GraphiteMultiSeries>> series) throws IOException {

		AtomicInteger counter = new AtomicInteger(0);

		Stream<GraphiteSeries> targetGroupsStream = series.stream().flatMap(s -> {
			return s.getValue().stream().map(g -> {
				g.getDatapoints().forEach(d->{
					d.setLabel(s.getKey());
				});
				return g;
			});
		});
		
		Map<String, List<GraphiteSeries>> targetGroups = targetGroupsStream.collect(Collectors.groupingBy(GraphiteSeries::getTarget));
		
		targetGroups.forEach((k,g) -> {
			File featuresFile = new File(this.baseDir, counter.get() + ".features.csv");
			File labelsFile = new File(this.baseDir, counter.getAndIncrement() + ".labels.csv");
			Stream<GraphiteDatapoint> datapointStream = g.stream().flatMap(d -> {
				return d.getDatapoints().stream();
			});
			try {
				FileWriter labelsWriter = new FileWriter(labelsFile);
				FileWriter featuresWriter = new FileWriter(featuresFile);
				// sort by epoch, and strip points where the value is valid
				List<GraphiteDatapoint> sortedDataPoints = datapointStream.sorted().filter(d -> d.getValue() != null && !Double.isNaN(d.getValue().doubleValue())).collect(Collectors.toList());
				AtomicLong dataPointCounter = new AtomicLong(sortedDataPoints.size());
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
			} catch (IOException e) {
				log.error("{}", e);
				throw new RuntimeException(e);
			}
		});
		
		return counter.get();
	}
}
