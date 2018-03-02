package net.savantly.learning.graphite.learners.timeseries;

import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
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
	private List<GraphiteQuery<String>> trainingQueries = new ArrayList<>();
	private List<INDArray> trainingData = new ArrayList<>();
	private RegressionNetwork network;

	private int epochs = 5;
	private double learningRate = 0.07;
	private int miniBatchSize = 100;
	
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
		if (this.trainingData.isEmpty()) {
			for (GraphiteQuery<String> graphiteQuery : trainingQueries) {
				List<INDArray> array = new GraphiteCsv(client.query(graphiteQuery)).asINDArray();
				this.trainingData.addAll(array);
			}
		}
		
		this.network = RegressionNetwork.builder()
				.setEpochs(epochs)
				.setLearningRate(learningRate)
				.setMiniBatchSize(miniBatchSize)
				.setTrainingData(trainingData)
				.build();
		
		return this;
	}
	
	public INDArray doTimeStep(int timeSteps) {
		return this.network.rnnTimeStep(timeSteps);
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

	public List<INDArray> getTrainingData() {
		return trainingData;
	}

	// Preload the data - this will prevent the queries from being executed
	public GraphitePredictor addTrainingData(INDArray... trainingData) {
		for (INDArray indArray : trainingData) {
			this.trainingData.add(indArray);
		}
		return this;
	}
	// Preload the data - this will prevent the queries from being executed
	public GraphitePredictor addTrainingData(Collection<INDArray> trainingData) {
		this.trainingData.addAll(trainingData);
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

}
