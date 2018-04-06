package net.savantly.learning.graphite.learners.timeseries;

import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.Function;

import org.datavec.api.writable.Writable;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.savantly.graphite.GraphiteClientFactory;
import net.savantly.graphite.QueryableGraphiteClient;
import net.savantly.graphite.query.GraphiteQuery;
import net.savantly.learning.graphite.domain.GraphiteCsv;
import net.savantly.learning.graphite.learners.RegressionNetwork;

public class GraphitePredictor {
	private static final Logger log = LoggerFactory.getLogger(GraphitePredictor.class);

	private QueryableGraphiteClient client;
	private GraphiteQuery<String> trainingQuery;
	private Function<String[], Boolean> dataFilter;
	private Collection<? extends Collection<? extends Collection<Writable>>> trainingData;
	private List<IterationListener> iterationListeners = new ArrayList<>();
	private RegressionNetwork network;

	private int epochs = 50;
	private double learningRate = 0.07;
	private double rnnLearningRate = 0.07;
	private int miniBatchSize = 1;
	private int windowSize;
	private int numOfInputs;
	private int hiddenLayerWidth = 10;
	
	public static GraphitePredictor builder() {
		return new GraphitePredictor();
	}
	
	public GraphitePredictor build() {
		return build(null);
	}
	
	public GraphitePredictor build(MultiLayerNetwork pretrainedNetwork) {
		if (this.client == null) {
			try {
				this.client = GraphiteClientFactory.queryableGraphiteClient("127.0.0.1");
			} catch (UnknownHostException | SocketException e) {
				log.error("{}", e);
				throw new RuntimeException(e);
			}
		}
		if (this.trainingData == null) {
			this.trainingData = new GraphiteCsv(client.query(this.trainingQuery), this.dataFilter).asRecords(windowSize);
		}
		
		this.network = RegressionNetwork.builder()
				.setEpochs(epochs)
				.setLearningRate(learningRate)
				.setRnnLearningRate(rnnLearningRate)
				.setTrainingData(trainingData)
				.setNumOfInputs(numOfInputs)
				.setHiddenLayerWidth(hiddenLayerWidth)
				.setIterationListeners(iterationListeners);
		if(pretrainedNetwork == null) {
			this.network = this.network.build();
		} else {
			this.network = this.network.build(pretrainedNetwork);
		}
		
		return this;
	}
	
	public float[] doTimeStep(int timeSteps, INDArray testingData) {
		return this.network.rnnTimeStep(timeSteps, testingData);
	}
	
	public INDArray predict(int timeSteps) {
		return this.network.predict(timeSteps);
	}
	
	public void update(INDArray input) {
		this.network.update(input);
	}

	public MultiLayerNetwork train() {
		return this.network.train();
	}

	public QueryableGraphiteClient getClient() {
		return client;
	}

	public GraphitePredictor setClient(QueryableGraphiteClient client) {
		this.client = client;
		return this;
	}

	public GraphiteQuery<String> getTrainingQuery() {
		return trainingQuery;
	}

	public GraphitePredictor setTrainingQuery(GraphiteQuery<String> trainingQuery) {
		this.trainingQuery = trainingQuery;
		return this;
	}

	public Collection<? extends Collection<? extends Collection<Writable>>> getTrainingData() {
		return trainingData;
	}

	// Preload the data - this will prevent the queries from being executed
	public GraphitePredictor setTrainingData(Collection<? extends Collection<? extends Collection<Writable>>> indArray) {
		this.trainingData = indArray;
		return this;
	}

	public int getEpochs() {
		return epochs;
	}

	public GraphitePredictor setEpochs(int iterations) {
		this.epochs = iterations;
		return this;
	}

	public int getMiniBatchSize() {
		return miniBatchSize;
	}

	public GraphitePredictor setMiniBatchSize(int miniBatchSize) {
		this.miniBatchSize = miniBatchSize;
		return this;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public GraphitePredictor setLearningRate(double learningRate) {
		this.learningRate = learningRate;
		return this;
	}

	public int getNumOfInputs() {
		return numOfInputs;
	}

	public GraphitePredictor setNumOfInputs(int numOfInputs) {
		this.numOfInputs = numOfInputs;
		return this;
	}

	public int getWindowSize() {
		return windowSize;
	}

	public GraphitePredictor setWindowSize(int windowSize) {
		this.windowSize = windowSize;
		return this;
	}

	public Function<String[], Boolean> getDataFilter() {
		return dataFilter;
	}

	public GraphitePredictor setDataFilter(Function<String[], Boolean> dataFilter) {
		this.dataFilter = dataFilter;
		return this;
	}

	public int getHiddenLayerWidth() {
		return hiddenLayerWidth;
	}

	public GraphitePredictor setHiddenLayerWidth(int hiddenLayerWidth) {
		this.hiddenLayerWidth = hiddenLayerWidth;
		return this;
	}
	
	public GraphitePredictor addIterationListener(IterationListener listener) {
		this.iterationListeners.add(listener);
		return this;
	}

	public double getRnnLearningRate() {
		return rnnLearningRate;
	}

	public GraphitePredictor setRnnLearningRate(double rnnLearningRate) {
		this.rnnLearningRate = rnnLearningRate;
		return this;
	}

}
