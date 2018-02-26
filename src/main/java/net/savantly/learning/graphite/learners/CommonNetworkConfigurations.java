package net.savantly.learning.graphite.learners;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class CommonNetworkConfigurations {

	public static MultiLayerConfiguration simpleClassification(int featureCount, int numberOfPossibleLabels, double learningRate) {
		int intermediateLayerWidth = featureCount * 10;
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(123) // Random number generator seed
				// for improved repeatability.
				// Optional.
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
				.weightInit(WeightInit.XAVIER).updater(Updater.NESTEROVS).learningRate(learningRate)
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue) // Not always required
				.gradientNormalizationThreshold(0.5).list()
				.layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(featureCount).nOut(intermediateLayerWidth).build())
				.layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
						.nIn(intermediateLayerWidth).nOut(numberOfPossibleLabels).build())
				.pretrain(false).backprop(true).build();
		return conf;
	}
}
