package net.savantly.learning.graphite.learners;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class MultiLayerLearnerBase implements MultiLayerLearner {

	private static final Logger log = LoggerFactory.getLogger(MultiLayerLearnerBase.class);
	private List<IterationListener> iterationListeners = new ArrayList<>();

	public MultiLayerConfiguration createNetworkConfiguration() {
		log.info("features: {}", this.getFeatureCount());
		return CommonNetworkConfigurations.simpleClassification(this.getFeatureCount(),
				this.getNumberOfPossibleLabels(), this.getLearningRate());
	}

	public MultiLayerNetwork train() {

		DataSetIterator trainData = getTrainingDataSets();
		DataSetIterator testData = getTestingDataSets();
		
		DataNormalization normalizer = this.getNormalizer();
		boolean normalize = (this.getNormalizer() != null);
		if (normalize) {
			normalizer.fit(trainData);		
			// Each DataSet
			// returned by 'trainData' iterator will be normalized
			trainData.setPreProcessor(normalizer);
			testData.setPreProcessor(normalizer);
		}

		// ----- Configure the network -----
		MultiLayerConfiguration conf = createNetworkConfiguration();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		net.setListeners(this.iterationListeners); // Print the score (loss function value) every 20 iterations

		// ----- Train the network, evaluating the test set performance at each epoch
		// -----
		int nEpochs = this.getNumberOfIterations();
		String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
		for (int i = 0; i < nEpochs; i++) {
			net.fit(trainData);

			if (!testData.hasNext()) {
				log.warn("there are no examples to test");
			} else {
				// Evaluate on the test set:
				Evaluation evaluation = net.evaluate(testData);
				log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));
			}

			testData.reset();
			trainData.reset();
		}

		log.info("----- completed training and testing -----");

		return net;
	}

	public abstract int getNumberOfPossibleLabels();

	public abstract int getNumberOfIterations();

	public abstract DataNormalization getNormalizer();

	public List<IterationListener> getIterationListeners() {
		return iterationListeners;
	}

	public abstract double getLearningRate();

	public abstract int getFeatureCount();
}
