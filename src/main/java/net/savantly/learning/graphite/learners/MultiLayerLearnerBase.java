package net.savantly.learning.graphite.learners;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class MultiLayerLearnerBase implements MultiLayerLearner {

	private static final Logger log = LoggerFactory.getLogger(MultiLayerLearnerBase.class);

	protected int numberOfPossibleLabels = 2;
	protected int numberOfIterations = 40;
	protected DataNormalization normalizer = new NormalizerStandardize();
	protected List<IterationListener> iterationListeners = new ArrayList<>();
	protected double learningRate = 0.005;
	protected int featureCount = 1;

	public MultiLayerConfiguration createNetworkConfiguration() {
		return CommonNetworkConfigurations.simpleClassification(featureCount, numberOfPossibleLabels, learningRate);
	}

	public MultiLayerNetwork train() {

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
}
