package net.savantly.learning.graphite.learners;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.primitives.Ints;

public class RegressionNetwork {
	private double learningRate = 0.07;
	private int epochs = 20;
	private List<INDArray> trainingData;
	private MultiLayerNetwork network;
	private int miniBatchSize;
	private int miniBatchIterations = 1;
	
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
		DataSetIterator iterator = getDataSetIterator(this.trainingData);
		//Train the network on the full data set, and evaluate in periodically
        for( int i=0; i<this.epochs; i++ ){
            iterator.reset();
            this.network.fit(iterator);
        }
		return this.network;
	}

	public INDArray rnnTimeStep(int timeSteps) {
		INDArray input = Nd4j.create(new int[] {timeSteps, getFeatureCount()});
		return this.network.rnnTimeStep(input);
	}
	
	public INDArray predict(int timeSteps) {
		INDArray input = Nd4j.create(new int[] {timeSteps, getFeatureCount()});
		List<INDArray> list = new ArrayList<>();
		list.add(input);
		return this.network.output(getDataSetIterator(list));
	}
	
	@SuppressWarnings({ "rawtypes", "unchecked" })
	public DataSetIterator getDataSetIterator(List<INDArray> data) {
        List<DataSet> listDs = new ArrayList<>();
		data.stream().forEach(d -> {
	        INDArray output = Nd4j.zeros(getOutputCount(), 1);
			DataSet ds = new DataSet(d, output);
			listDs.add(ds);
		});
        return new ListDataSetIterator(listDs, this.miniBatchSize);
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
			return t.size(1);
		}).max(Ints::compare).get();
	}

	public List<INDArray> getTrainingData() {
		return trainingData;
	}

	public RegressionNetwork setTrainingData(List<INDArray> trainingData) {
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
		return this.trainingData.size();
	}

	public int getMiniBatchSize() {
		return miniBatchSize;
	}

	public RegressionNetwork setMiniBatchSize(int miniBatchSize) {
		this.miniBatchSize = miniBatchSize;
		return this;
	}
}
