package net.savantly.learning.graphite.learners;

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
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.savantly.learning.graphite.convert.GraphiteToCsv;
import net.savantly.learning.graphite.domain.GraphiteMultiSeries;

public class SimpleStateClassifier {
	private static final Logger log = LoggerFactory.getLogger(SimpleStateClassifier.class);
	
	private final int numberOfPossibleLabels = 2;
	private List<GraphiteMultiSeries> positiveTrainingExamples = new ArrayList<>();
	private List<GraphiteMultiSeries> negativeTrainingExamples = new ArrayList<>();
	private List<GraphiteMultiSeries> positiveTestingExamples = new ArrayList<>();
	private List<GraphiteMultiSeries> negativeTestingExamples = new ArrayList<>();
	private File workingDirectory = new File("./data");
	private int miniBatchSize = 10;
	private boolean doRegression = false;
	private int numberOfIterations = 40;

	private SimpleStateClassifier() {
	}

	public static SimpleStateClassifier builder() {
		return new SimpleStateClassifier();
	}

	public TrainingResult train() throws IOException, InterruptedException {
		TrainingResult result = new TrainingResult();
		
		List<Pair<String, GraphiteMultiSeries>> pairs = new ArrayList<>();

		// Add the positive condition examples
		this.positiveTrainingExamples.stream().forEach(g -> {
			Pair<String, GraphiteMultiSeries> p = Pair.of("0", g);
			pairs.add(p);
		});

		// Add the negative condition examples
		this.negativeTrainingExamples.stream().forEach(g -> {
			Pair<String, GraphiteMultiSeries> p = Pair.of("0", g);
			pairs.add(p);
		});

		// Write the transformed data to csv files in the dir
		int fileCount = GraphiteToCsv.get(this.workingDirectory.getAbsolutePath()).createFileSequence(pairs);

		Arrays.stream(this.workingDirectory.list()).forEach(f -> {
			log.info(f);
		});

		String featureFilesMatcher = this.workingDirectory.toPath().resolve("%d.features.csv").toAbsolutePath().toString();
		String labelFilesMatcher = this.workingDirectory.toPath().resolve("%d.labels.csv").toAbsolutePath().toString();

		// ----- Load the training data -----
		SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(0, ",");

		trainFeatures.initialize(new NumberedFileInputSplit(featureFilesMatcher, 0, fileCount - 2));
		SequenceRecordReader trainLabels = new CSVSequenceRecordReader(0, ",");
		trainLabels.initialize(new NumberedFileInputSplit(labelFilesMatcher, 0, fileCount - 2));

		int miniBatchSize = 10;
		int numberOfPossibleLabels = 2;
		boolean doRegression = false;

		DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize,
				numberOfPossibleLabels, doRegression, AlignmentMode.ALIGN_END);

		// Normalize the training data
		DataNormalization normalizer = new NormalizerStandardize();
		normalizer.fit(trainData); // Collect training data statistics
		trainData.reset();

		// Use previously collected statistics to normalize on-the-fly. Each DataSet
		// returned by 'trainData' iterator will be normalized
		trainData.setPreProcessor(normalizer);

		// ----- Load the test data -----
		// Same process as for the training data.
		SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
		testFeatures.initialize(new NumberedFileInputSplit(featureFilesMatcher, 0, fileCount - 1));
		SequenceRecordReader testLabels = new CSVSequenceRecordReader();
		testLabels.initialize(new NumberedFileInputSplit(labelFilesMatcher, 0, fileCount - 1));

		DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize,
				numberOfPossibleLabels, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

		// testData.setPreProcessor(normalizer); // Note that we are using the exact
		// same normalization process as the
		// training data

		// ----- Configure the network -----
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(123) // Random number generator seed
																						// for improved repeatability.
																						// Optional.
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
				.weightInit(WeightInit.XAVIER).updater(Updater.NESTEROVS).learningRate(0.005)
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue) // Not always required, but
																							// helps with this data set
				.gradientNormalizationThreshold(0.5).list()
				.layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build())
				.layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
						.nIn(10).nOut(numberOfPossibleLabels).build())
				.pretrain(false).backprop(true).build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		net.setListeners(new ScoreIterationListener(20)); // Print the score (loss function value) every 20 iterations

		// ----- Train the network, evaluating the test set performance at each epoch
		// -----
		int nEpochs = 40;
		String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
		for (int i = 0; i < nEpochs; i++) {
			net.fit(trainData);

			// Evaluate on the test set:
			Evaluation evaluation = net.evaluate(testData);
			log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));

			testData.reset();
			trainData.reset();
		}

		log.info("----- Example Complete -----");

		return result;
	}

	public List<GraphiteMultiSeries> getPositiveTrainingExamples() {
		return positiveTrainingExamples;
	}

	public SimpleStateClassifier setPositiveTrainingExamples(List<GraphiteMultiSeries> positiveTrainingExamples) {
		this.positiveTrainingExamples = positiveTrainingExamples;
		return this;
	}

	public List<GraphiteMultiSeries> getNegativeTrainingExamples() {
		return negativeTrainingExamples;
	}

	public SimpleStateClassifier setNegativeTrainingExamples(List<GraphiteMultiSeries> negativeTrainingExamples) {
		this.negativeTrainingExamples = negativeTrainingExamples;
		return this;
	}

	public List<GraphiteMultiSeries> getPositiveTestingExamples() {
		return positiveTestingExamples;
	}

	public void setPositiveTestingExamples(List<GraphiteMultiSeries> positiveTestingExamples) {
		this.positiveTestingExamples = positiveTestingExamples;
	}

	public List<GraphiteMultiSeries> getNegativeTestingExamples() {
		return negativeTestingExamples;
	}

	public void setNegativeTestingExamples(List<GraphiteMultiSeries> negativeTestingExamples) {
		this.negativeTestingExamples = negativeTestingExamples;
	}

	public File getWorkingDirectory() {
		return workingDirectory;
	}

	public SimpleStateClassifier setWorkingDirectory(File workingDirectory) {
		this.workingDirectory = workingDirectory;
		return this;
	}

	public int getMiniBatchSize() {
		return miniBatchSize;
	}

	public SimpleStateClassifier setMiniBatchSize(int miniBatchSize) {
		this.miniBatchSize = miniBatchSize;
		return this;
	}

	public boolean isDoRegression() {
		return doRegression;
	}

	public SimpleStateClassifier setDoRegression(boolean doRegression) {
		this.doRegression = doRegression;
		return this;
	}

	public int getNumberOfIterations() {
		return numberOfIterations;
	}

	public void setNumberOfIterations(int numberOfIterations) {
		this.numberOfIterations = numberOfIterations;
	}

	class TrainingResult {
		MultiLayerNetwork network;
		Evaluation evaluation;
	}

}
