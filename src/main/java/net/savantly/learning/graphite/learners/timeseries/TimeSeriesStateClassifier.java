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
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.savantly.learning.graphite.convert.GraphiteToCsv;
import net.savantly.learning.graphite.convert.GraphiteToCsv.CsvResult;
import net.savantly.learning.graphite.domain.GraphiteMultiSeries;

public class TimeSeriesStateClassifier {
	private static final Logger log = LoggerFactory.getLogger(TimeSeriesStateClassifier.class);

	private final int numberOfPossibleLabels = 2;
	private List<GraphiteMultiSeries> positiveExamples = new ArrayList<>();
	private List<GraphiteMultiSeries> negativeExamples = new ArrayList<>();
	private File workingDirectory = new File("./data");
	private int miniBatchSize = 10;
	private boolean doRegression = false;
	private int numberOfIterations = 40;
	private boolean isBuilt = false;
	private CsvResult filePairCount;
	private DataNormalization normalizer = new NormalizerStandardize();
	private List<IterationListener> iterationListeners = new ArrayList<>();
	private double learningRate = 0.005;

	private TimeSeriesStateClassifier() {
	}

	public static TimeSeriesStateClassifier builder() {
		return new TimeSeriesStateClassifier();
	}

	public TimeSeriesStateClassifier build() throws IOException {
		this.filePairCount = createCsvFiles();
		this.isBuilt = true;
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

	public MultiLayerNetwork train() throws IOException, InterruptedException {
		if (!this.isBuilt) {
			throw new RuntimeException("must call build() first");
		}

		DataSetIterator trainData = getTrainingDataSets();

		// Normalize the training data
		normalizer.fit(trainData); // Collect training data statistics
		trainData.reset();

		// Use previously collected statistics to normalize on-the-fly. Each DataSet
		// returned by 'trainData' iterator will be normalized
		trainData.setPreProcessor(normalizer);

		DataSetIterator testData = getTestingDataSets();
		testData.setPreProcessor(normalizer);

		// ----- Configure the network -----
		MultiLayerConfiguration conf = createNetworkConfiguration();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		net.setListeners(this.iterationListeners); // Print the score (loss function value) every 20 iterations

		// ----- Train the network, evaluating the test set performance at each epoch
		// -----
		int nEpochs = this.numberOfIterations;
		String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
		for (int i = 0; i < nEpochs; i++) {
			net.fit(trainData);

			// Evaluate on the test set:
			Evaluation evaluation = net.evaluate(testData);
			log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));

			testData.reset();
			trainData.reset();
		}

		log.info("----- completed training and testing -----");

		return net;
	}

	private MultiLayerConfiguration createNetworkConfiguration() {
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(123) // Random number generator seed
				// for improved repeatability.
				// Optional.
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
				.weightInit(WeightInit.XAVIER).updater(Updater.NESTEROVS).learningRate(this.learningRate)
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue) // Not always required, but
				// helps with this data set
				.gradientNormalizationThreshold(0.5).list()
				.layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build())
				.layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
						.nIn(10).nOut(numberOfPossibleLabels).build())
				.pretrain(false).backprop(true).build();
		return conf;
	}

	private DataSetIterator getTestingDataSets() throws IOException, InterruptedException {
		String featuresFilePattern = this.workingDirectory.toPath().resolve("%d.train.features.csv").toAbsolutePath()
				.toString();
		String labelsFilePattern = this.workingDirectory.toPath().resolve("%d.train.labels.csv").toAbsolutePath()
				.toString();

		// ----- Load the test data -----
		// Same process as for the training data.
		SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
		testFeatures.initialize(new NumberedFileInputSplit(featuresFilePattern, 1, this.filePairCount.getTrainFileCount()));
		SequenceRecordReader testLabels = new CSVSequenceRecordReader();
		testLabels.initialize(new NumberedFileInputSplit(labelsFilePattern, 1, this.filePairCount.getTrainFileCount()));

		DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize,
				numberOfPossibleLabels, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
		return testData;
	}

	private DataSetIterator getTrainingDataSets() throws IOException, InterruptedException {
		String featuresFilePattern = this.workingDirectory.toPath().resolve("%d.test.features.csv").toAbsolutePath()
				.toString();
		String labelsFilePattern = this.workingDirectory.toPath().resolve("%d.test.labels.csv").toAbsolutePath()
				.toString();

		// ----- Load the training data -----
		SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();

		trainFeatures.initialize(new NumberedFileInputSplit(featuresFilePattern, 1, this.filePairCount.getTestFileCount()));
		
		SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
		trainLabels.initialize(new NumberedFileInputSplit(labelsFilePattern, 1, this.filePairCount.getTestFileCount()));

		DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize,
				numberOfPossibleLabels, doRegression, AlignmentMode.ALIGN_END);
		return trainData;
	}

	public List<GraphiteMultiSeries> getPostiveExamples() {
		return positiveExamples;
	}

	public TimeSeriesStateClassifier setPositiveExamples(List<GraphiteMultiSeries> positiveTrainingExamples) {
		this.positiveExamples = positiveTrainingExamples;
		return this;
	}

	public List<GraphiteMultiSeries> getNegativeExamples() {
		return negativeExamples;
	}

	public TimeSeriesStateClassifier setNegativeExamples(List<GraphiteMultiSeries> negativeTrainingExamples) {
		this.negativeExamples = negativeTrainingExamples;
		return this;
	}

	public File getWorkingDirectory() {
		return workingDirectory;
	}

	public TimeSeriesStateClassifier setWorkingDirectory(File workingDirectory) {
		this.workingDirectory = workingDirectory;
		return this;
	}

	public int getMiniBatchSize() {
		return miniBatchSize;
	}

	public TimeSeriesStateClassifier setMiniBatchSize(int miniBatchSize) {
		this.miniBatchSize = miniBatchSize;
		return this;
	}

	public boolean isDoRegression() {
		return doRegression;
	}

	public TimeSeriesStateClassifier setDoRegression(boolean doRegression) {
		this.doRegression = doRegression;
		return this;
	}

	public int getNumberOfIterations() {
		return numberOfIterations;
	}

	public TimeSeriesStateClassifier setNumberOfIterations(int numberOfIterations) {
		this.numberOfIterations = numberOfIterations;
		return this;
	}

	public DataNormalization getNormalizer() {
		return this.normalizer;
	}

	public TimeSeriesStateClassifier setNormalizer(DataNormalization normalizer) {
		this.normalizer = normalizer;
		return this;
	}

	public List<IterationListener> getIterationListeners() {
		return iterationListeners;
	}
	public TimeSeriesStateClassifier setIterationListeners(List<IterationListener> iterationListeners) {
		this.iterationListeners = iterationListeners;
		return this;
	}
	public TimeSeriesStateClassifier addIterationListener(IterationListener listener) {
		this.iterationListeners.add(listener);
		return this;
	}
	
	public double getLearningRate() {
		return this.learningRate;
	}
	public TimeSeriesStateClassifier setLearningRate(double learningRate) {
		this.learningRate = learningRate;
		return this;
	}

}
