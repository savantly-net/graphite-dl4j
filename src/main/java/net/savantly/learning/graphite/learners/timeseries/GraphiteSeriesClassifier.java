package net.savantly.learning.graphite.learners.timeseries;

import java.io.File;
import java.io.IOException;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator.AlignmentMode;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.savantly.graphite.GraphiteClientFactory;
import net.savantly.graphite.QueryableGraphiteClient;
import net.savantly.graphite.query.GraphiteQuery;
import net.savantly.learning.graphite.domain.GraphiteCsv;
import net.savantly.learning.graphite.learners.MultiLayerLearnerBase;
import net.savantly.learning.graphite.sequence.GraphiteSequenceRecordReader;

public class GraphiteSeriesClassifier extends MultiLayerLearnerBase {
	private static final Logger log = LoggerFactory.getLogger(GraphiteSeriesClassifier.class);

	private QueryableGraphiteClient client;
	private List<GraphiteQuery<String>> positiveExamples = new ArrayList<>();
	private List<GraphiteQuery<String>> negativeExamples = new ArrayList<>();
	private File workingDirectory;
	private int miniBatchSize = 1;
	private double percentTrain = 0.75;
	private List<Pair<INDArray, INDArray>> trainingData;
	private List<Pair<INDArray, INDArray>> testingData;
	private DataSet allData;
	private double learningRate = 0.007;
	private List<IterationListener> iterationListeners;
	private int numberOfIterations;
	private DataNormalization normalizer;
	private int featureCount;

	private GraphiteSeriesClassifier() { }

	public static GraphiteSeriesClassifier builder() {
		return new GraphiteSeriesClassifier();
	}

	public GraphiteSeriesClassifier build() throws IOException {
		if (this.workingDirectory == null) {
			this.workingDirectory = Files.createDirectories(Paths.get("data")).toFile();
		}
		
		if (this.client == null) {
			try {
				this.client = GraphiteClientFactory.queryableGraphiteClient("127.0.0.1");
			} catch (UnknownHostException | SocketException e) {
				log.error("{}", e);
				throw new RuntimeException(e);
			}
		}
		if (this.allData == null) {
			this.allData = this.buildDataFromQueries();
		}

		return this;
	}
	
	private DataSet buildDataFromQueries() {
		Pair<Boolean, GraphiteQuery<String>>[] multiSeries = 
				new Pair[this.negativeExamples.size() + this.positiveExamples.size()];
		
		final AtomicInteger pairCounter = new AtomicInteger(0);
		this.negativeExamples.forEach(e -> {
			multiSeries[pairCounter.getAndIncrement()] = Pair.of(false, e);
		});
		this.positiveExamples.forEach(e -> {
			multiSeries[pairCounter.getAndIncrement()] = Pair.of(true, e);
		});
		ArrayList<Pair<INDArray, INDArray>> resultPairs = new ArrayList<Pair<INDArray, INDArray>>();
		Arrays.stream(multiSeries).forEach(s -> {
			List<DataSet> data = new GraphiteCsv(client.query(s.getRight())).asLabeledDataSet(s.getLeft());
			resultPairs.add(Pair.of(data.get(0).getFeatures(), data.get(0).getLabels()));
		});
		
		double trainingCount = this.percentTrain * resultPairs.size();
		// TODO: FINISH implementation - set training and testing data
		
		return null;
	}
	
	public DataSetIterator createDataSetIterator(List<Pair<INDArray, INDArray>> dataPairs) {

		SequenceRecordReader featuresReader = new GraphiteSequenceRecordReader(dataPairs.stream().map(p -> {
			return p.getFirst();
		}).collect(Collectors.toList()));
		
		SequenceRecordReader labelsReader = new GraphiteSequenceRecordReader(dataPairs.stream().map(p -> {
			return p.getSecond();
		}).collect(Collectors.toList()));
		
		SequenceRecordReaderDataSetIterator iterator = 
				new SequenceRecordReaderDataSetIterator(
						featuresReader, labelsReader, miniBatchSize, this.getNumberOfPossibleLabels(), false, AlignmentMode.ALIGN_END);
		return iterator;
	}

	public DataSetIterator getTestingDataSets() {
		return createDataSetIterator(this.testingData);
	}

	public DataSetIterator getTrainingDataSets() {
		return createDataSetIterator(this.trainingData);
	}

	public List<GraphiteQuery<String>> getPostiveExamples() {
		return positiveExamples;
	}

	public GraphiteSeriesClassifier setPositiveExamples(List<GraphiteQuery<String>> positiveTrainingExamples) {
		this.positiveExamples = positiveTrainingExamples;
		return this;
	}

	public List<GraphiteQuery<String>> getNegativeExamples() {
		return negativeExamples;
	}

	public GraphiteSeriesClassifier setNegativeExamples(List<GraphiteQuery<String>> negativeTrainingExamples) {
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

	@Override
	public int getNumberOfPossibleLabels() {
		return 2;
	}

	@Override
	public int getFeatureCount() {
		return this.featureCount;
	}

}
