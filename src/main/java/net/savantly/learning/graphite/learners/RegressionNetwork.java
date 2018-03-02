package net.savantly.learning.graphite.learners;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.primitives.Ints;

public class RegressionNetwork {
	private static final Logger log = LoggerFactory.getLogger(RegressionNetwork.class);
	private double learningRate = 0.07;
	private int epochs = 20;
	private List<DataSet> trainingData;
	private MultiLayerNetwork network;
	private int miniBatchSize = 1;
	private int miniBatchIterations = 1;
	private double percentTrain = 0.75;
	
	protected RegressionNetwork() {}
	
	public static RegressionNetwork builder() {
		return new RegressionNetwork();
	}
	
	public RegressionNetwork build() {
		if (this.trainingData == null) {
			throw new RuntimeException("no training data found");
		}
		this.network = new MultiLayerNetwork(getNetworkConfiguration());
		this.network.init();
		this.network.setListeners(new IterationListener[]{
				new ScoreIterationListener(20),
				new PerformanceListener(10)});
		return this;
	}
	
	public MultiLayerNetwork train() {
		List<DataSet> training = new ArrayList<>();
		List<DataSet> testing = new ArrayList<>();
		this.trainingData.stream().forEach(d->{
			SplitTestAndTrain splitData = d.splitTestAndTrain(percentTrain);
			training.add(splitData.getTrain());
			testing.add(splitData.getTest());
		});
		DataSetIterator trainIterator = getDataSetIterator(training);
		DataSetIterator testIterator = getDataSetIterator(testing);
		
		DataNormalization normalizer = new NormalizerStandardize();
		normalizer.fit(trainIterator);
		trainIterator.setPreProcessor(normalizer);
		testIterator.setPreProcessor(normalizer);
		
		//Train the network on the full data set, and evaluate in periodically
		String evalStringFormat = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
        for( int i=0; i<this.epochs; i++ ){
            trainIterator.reset();
            this.network.fit(trainIterator);
            
            if (!testIterator.hasNext()) {
				log.warn("there are no examples to test");
			} else {
				// Evaluate on the test set:
				Evaluation evaluation = this.network.evaluate(testIterator);
				log.info(String.format(evalStringFormat, i, evaluation.accuracy(), evaluation.f1()));
			}

            testIterator.reset();
            trainIterator.reset();
        }
		return this.network;
	}

	public INDArray rnnTimeStep(int timeSteps) {
		INDArray input = Nd4j.create(new int[] {timeSteps, getFeatureCount()});
		return this.network.rnnTimeStep(input);
	}
	
	public INDArray predict(int timeSteps) {
		INDArray input = Nd4j.create(new int[] {timeSteps, getFeatureCount()});
		return this.network.output(input);
	}
	
	@SuppressWarnings({ "rawtypes", "unchecked" })
	public DataSetIterator getDataSetIterator(List<DataSet> ds) {
        return new ListDataSetIterator(ds, this.miniBatchSize);
	}

	public MultiLayerConfiguration getNetworkConfiguration() {
		return CommonNetworkConfigurations.recurrentNetwork(getFeatureCount(), getOutputCount(), getLearningRate(), getMiniBatchIterations());
	}

	private int getMiniBatchIterations() {
		return this.miniBatchIterations ;
	}

	private double getLearningRate() {
		return this.learningRate;
	}
	public RegressionNetwork setLearningRate(double rate) {
		this.learningRate = rate;
		return this;
	}

	public int getFeatureCount() {
		return this.trainingData.stream().map(t -> {
			return t.getFeatures().size(1);
		}).max(Ints::compare).get();
	}

	public List<DataSet> getTrainingData() {
		return trainingData;
	}

	public RegressionNetwork setTrainingData(List<DataSet> trainingData) {
		this.trainingData = trainingData;
		return this;
	}

	public int getEpochs() {
		return epochs;
	}

	public RegressionNetwork setEpochs(int iterations) {
		this.epochs = iterations;
		return this;
	}

	public int getOutputCount() {
		return this.getFeatureCount();
	}

	public int getMiniBatchSize() {
		return miniBatchSize;
	}

	public RegressionNetwork setMiniBatchSize(int miniBatchSize) {
		this.miniBatchSize = miniBatchSize;
		return this;
	}
}
