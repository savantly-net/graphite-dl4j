package net.savantly.learning.graphite.learners.timeseries;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator.AlignmentMode;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.savantly.learning.graphite.convert.GraphiteToCsv;
import net.savantly.learning.graphite.convert.GraphiteToCsv.CsvResult;
import net.savantly.learning.graphite.domain.GraphiteMultiSeries;
import net.savantly.learning.graphite.learners.MultiLayerLearnerBase;

public class GraphiteSeriesClassifier extends MultiLayerLearnerBase {
	private static final Logger log = LoggerFactory.getLogger(GraphiteSeriesClassifier.class);

	private List<GraphiteMultiSeries> positiveExamples = new ArrayList<>();
	private List<GraphiteMultiSeries> negativeExamples = new ArrayList<>();
	private File workingDirectory = new File("./data");
	private int miniBatchSize = 10;
	private boolean doRegression = false;
	private CsvResult filePairCount;

	private GraphiteSeriesClassifier() { }

	public static GraphiteSeriesClassifier builder() {
		return new GraphiteSeriesClassifier();
	}

	public GraphiteSeriesClassifier build() throws IOException {
		this.filePairCount = createCsvFiles();
		return this;
	}

	private CsvResult createCsvFiles() throws IOException {
		List<Pair<String, GraphiteMultiSeries>> pairs = new ArrayList<>();

		// Add the positive condition examples
		this.positiveExamples.stream().forEach(g -> {
			Pair<String, GraphiteMultiSeries> p = Pair.of("0", g);
			pairs.add(p);
		});

		// Add the negative condition examples
		this.negativeExamples.stream().forEach(g -> {
			Pair<String, GraphiteMultiSeries> p = Pair.of("0", g);
			pairs.add(p);
		});

		// Write the transformed data to csv files in the dir
		CsvResult fileCounts = GraphiteToCsv.get(this.workingDirectory.getAbsolutePath()).createFileSequence(pairs);

		Arrays.stream(this.workingDirectory.list()).forEach(f -> {
			log.info(f);
		});
		return fileCounts;
	}

	public DataSetIterator getTestingDataSets() {
		try {
			String featuresFilePattern = this.workingDirectory.toPath().resolve("%d.train.features.csv")
					.toAbsolutePath().toString();
			String labelsFilePattern = this.workingDirectory.toPath().resolve("%d.train.labels.csv").toAbsolutePath()
					.toString();

			// ----- Load the test data -----
			// Same process as for the training data.
			SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
			testFeatures.initialize(
					new NumberedFileInputSplit(featuresFilePattern, 1, this.filePairCount.getTrainFileCount()));
			SequenceRecordReader testLabels = new CSVSequenceRecordReader();
			testLabels.initialize(
					new NumberedFileInputSplit(labelsFilePattern, 1, this.filePairCount.getTrainFileCount()));

			DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize,
					numberOfPossibleLabels, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
			return testData;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public DataSetIterator getTrainingDataSets() {
		try {
			String featuresFilePattern = this.workingDirectory.toPath().resolve("%d.test.features.csv").toAbsolutePath()
					.toString();
			String labelsFilePattern = this.workingDirectory.toPath().resolve("%d.test.labels.csv").toAbsolutePath()
					.toString();

			// ----- Load the training data -----
			SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();

			trainFeatures.initialize(
					new NumberedFileInputSplit(featuresFilePattern, 1, this.filePairCount.getTestFileCount()));

			SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
			trainLabels.initialize(
					new NumberedFileInputSplit(labelsFilePattern, 1, this.filePairCount.getTestFileCount()));

			DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels,
					miniBatchSize, numberOfPossibleLabels, doRegression, AlignmentMode.ALIGN_END);
			return trainData;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public List<GraphiteMultiSeries> getPostiveExamples() {
		return positiveExamples;
	}

	public GraphiteSeriesClassifier setPositiveExamples(List<GraphiteMultiSeries> positiveTrainingExamples) {
		this.positiveExamples = positiveTrainingExamples;
		return this;
	}

	public List<GraphiteMultiSeries> getNegativeExamples() {
		return negativeExamples;
	}

	public GraphiteSeriesClassifier setNegativeExamples(List<GraphiteMultiSeries> negativeTrainingExamples) {
		this.negativeExamples = negativeTrainingExamples;
		return this;
	}

	public File getWorkingDirectory() {
		return workingDirectory;
	}

	public GraphiteSeriesClassifier setWorkingDirectory(File workingDirectory) {
		this.workingDirectory = workingDirectory;
		return this;
	}

	public int getMiniBatchSize() {
		return miniBatchSize;
	}

	public GraphiteSeriesClassifier setMiniBatchSize(int miniBatchSize) {
		this.miniBatchSize = miniBatchSize;
		return this;
	}

	public boolean isDoRegression() {
		return doRegression;
	}

	public GraphiteSeriesClassifier setDoRegression(boolean doRegression) {
		this.doRegression = doRegression;
		return this;
	}

	public int getNumberOfIterations() {
		return numberOfIterations;
	}

	public GraphiteSeriesClassifier setNumberOfIterations(int numberOfIterations) {
		this.numberOfIterations = numberOfIterations;
		return this;
	}

	public DataNormalization getNormalizer() {
		return this.normalizer;
	}

	public GraphiteSeriesClassifier setNormalizer(DataNormalization normalizer) {
		this.normalizer = normalizer;
		return this;
	}

	public List<IterationListener> getIterationListeners() {
		return iterationListeners;
	}

	public GraphiteSeriesClassifier setIterationListeners(List<IterationListener> iterationListeners) {
		this.iterationListeners = iterationListeners;
		return this;
	}

	public GraphiteSeriesClassifier addIterationListener(IterationListener listener) {
		this.iterationListeners.add(listener);
		return this;
	}


	public double getLearningRate() {
		return this.learningRate;
	}

	public GraphiteSeriesClassifier setLearningRate(double learningRate) {
		this.learningRate = learningRate;
		return this;
	}

}
