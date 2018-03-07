package net.savantly.learning.graphite.learners.timeseries;

import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.datavec.api.writable.Writable;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
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
	private List<GraphiteQuery<String>> trainingQueries = new ArrayList<>();
	private Collection<? extends Collection<? extends Collection<Writable>>> trainingData;
	private RegressionNetwork network;

	private int epochs = 20;
	private double learningRate = 0.07;
	private int miniBatchSize = 1;

	private int windowSize;

	private int numOfInputs;
	
	public static GraphitePredictor builder() {
		return new GraphitePredictor();
	}
	
	public GraphitePredictor build() {
		if (this.client == null) {
			try {
				this.client = GraphiteClientFactory.queryableGraphiteClient("127.0.0.1");
			} catch (UnknownHostException | SocketException e) {
				log.error("{}", e);
				throw new RuntimeException(e);
			}
		}
		if (this.trainingData == null) {
			List<DataSet> dsList = new ArrayList<>();
			for (GraphiteQuery<String> graphiteQuery : trainingQueries) {
				List<DataSet> ds = new GraphiteCsv(client.query(graphiteQuery)).asDataSetLagWindow(windowSize);
				dsList.addAll(ds);
			}
		}
		
		this.network = RegressionNetwork.builder()
				.setEpochs(epochs)
				.setLearningRate(learningRate)
				.setTrainingData(trainingData)
				.setNumOfInputs(numOfInputs)
				.build();
		
		return this;
	}
	
	public float[] doTimeStep(int timeSteps, INDArray testingData) {
		return this.network.rnnTimeStep(timeSteps, testingData);
	}
	
	public INDArray predict(int timeSteps) {
		return this.network.predict(timeSteps);
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

	public List<GraphiteQuery<String>> getTrainingQueries() {
		return trainingQueries;
	}

	public GraphitePredictor setTrainingQueries(List<GraphiteQuery<String>> trainingQueries) {
		this.trainingQueries = trainingQueries;
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

}
