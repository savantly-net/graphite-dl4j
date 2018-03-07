package net.savantly.learning.graphite.learners;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class CommonNetworkConfigurations {

	// SUCKS
	public static MultiLayerConfiguration defaultNetwork(int numInputs, int numOutputs, double learningRate) {
		int intermediateLayerWidth = numInputs * 10;
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
				.weightInit(WeightInit.XAVIER).updater(Updater.NESTEROVS)
				.learningRate(learningRate)
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue) // Not always required
				.gradientNormalizationThreshold(0.5)
				.list()
				.layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(numInputs).nOut(intermediateLayerWidth).build())
				.layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.activation(Activation.SOFTMAX)
						.nIn(intermediateLayerWidth)
						.nOut(numOutputs).build())
				.pretrain(false).backprop(true).build();
		return conf;
	}
	
	// SUCKS
	public static MultiLayerConfiguration simpleClassification(int numInputs, int numOutputs, double learningRate) {
		int intermediateLayerWidth = numInputs * 10;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.RELU)
                .list()
                .layer(0, new GravesLSTM.Builder()
                		.nIn(numInputs)
                		.nOut(intermediateLayerWidth)
                		.activation(Activation.SOFTSIGN).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                		.nIn(intermediateLayerWidth)
                		.nOut(numOutputs)
                		.activation(Activation.SOFTMAX).build())
                .build();
		return conf;
	}
	
	// SUCKS
	public static MultiLayerConfiguration simpleRegression(int featureCount, double learningRate, int miniBatchIterations, int numOutputs) {
		int hiddenLayerWidth = featureCount * 2;
		int seed = 1234;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(miniBatchIterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS)     //To configure: .updater(new Nesterovs(0.9))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(featureCount).nOut(hiddenLayerWidth)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(hiddenLayerWidth).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();
		return conf;
	}
	
	public static MultiLayerConfiguration recurrentNetwork(int numInputs, int hiddenLayerWidth, int numOutputs, double learningRate, int miniBatchIterations) {
        int tbpttLength = 500;
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(miniBatchIterations)
                .learningRate(learningRate)
                .seed(12345)
                .regularization(true)
                	.l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
                .list()
                    .layer(0, new GravesLSTM.Builder().nIn(numInputs).nOut(hiddenLayerWidth)
                                .activation(Activation.SOFTSIGN).build())
                    .layer(1, new RnnOutputLayer.Builder(LossFunction.MSE).activation(Activation.IDENTITY)        //MCXENT + softmax for classification
                                .nIn(hiddenLayerWidth).nOut(numOutputs).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                .pretrain(false).backprop(true)
                .build();
		return conf;
	}
}
