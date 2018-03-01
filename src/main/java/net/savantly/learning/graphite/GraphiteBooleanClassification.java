package net.savantly.learning.graphite;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.assertj.core.util.Arrays;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.JsonNode;

import net.savantly.graphite.GraphiteClient;
import net.savantly.graphite.GraphiteClientFactory;
import net.savantly.graphite.QueryableGraphiteClient;
import net.savantly.graphite.query.GraphiteQuery;
import net.savantly.learning.graphite.convert.GraphiteToDataSet;
import net.savantly.learning.graphite.domain.GraphiteMultiSeries;
import net.savantly.learning.graphite.learners.timeseries.GraphiteSeriesClassifier;

public class GraphiteBooleanClassification {
	
	private static final Logger log = LoggerFactory.getLogger(GraphiteBooleanClassification.class);

	private List<GraphiteMultiSeries> positiveTrainingExamples = new ArrayList<>();
	private List<GraphiteMultiSeries> negativeTrainingExamples = new ArrayList<>();
	private List<GraphiteQuery<JsonNode>> positiveQueries = new ArrayList<>();
	private List<GraphiteQuery<JsonNode>> negativeQueries = new ArrayList<>();
	private GraphiteSeriesClassifier classifier = GraphiteSeriesClassifier.builder();
	private File workingDirectory;
	private QueryableGraphiteClient client;
	private int numberOfIterations = 40;
	private MultiLayerNetwork network;

	private GraphiteBooleanClassification() {
	}

	public static GraphiteBooleanClassification builder() {
		return new GraphiteBooleanClassification();
	}

	public GraphiteBooleanClassification build() throws IOException {
		if (this.client == null) {
			this.client = GraphiteClientFactory.queryableGraphiteClient("127.0.0.1");
		}
		for (GraphiteQuery<JsonNode> graphiteQuery : negativeQueries) {
			this.negativeTrainingExamples.add(GraphiteMultiSeries.from(client.query(graphiteQuery)));
		}
		for (GraphiteQuery<JsonNode> graphiteQuery : positiveQueries) {
			this.positiveTrainingExamples.add(GraphiteMultiSeries.from(client.query(graphiteQuery)));
		}
		this.classifier.setWorkingDirectory(workingDirectory).setNumberOfIterations(numberOfIterations)
				.setNegativeExamples(negativeTrainingExamples).setPositiveExamples(positiveTrainingExamples).build();
		return this;
	}

	public MultiLayerNetwork train() {
		this.network = this.classifier.train();
		return this.network;
	}

	public INDArray evaluate(GraphiteQuery<JsonNode> query) {
		
		GraphiteMultiSeries multiSeries = GraphiteMultiSeries.from(this.client.query(query));
		
		List<Pair<INDArray, INDArray>> dataPairs = GraphiteToDataSet
				.toTimeSeriesNDArray(Arrays.array(Pair.of(0, multiSeries)));

		DataSetIterator dsIterator = classifier.createDataSetIterator(dataPairs);

	    INDArray timeSeriesOutput = this.network.output(dsIterator);
	    log.debug(timeSeriesOutput.toString());
	    int timeSeriesLength = timeSeriesOutput.size(2);		//Size of time dimension
	    INDArray lastTimeStepProbabilities = timeSeriesOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength-1));
	    log.debug(lastTimeStepProbabilities.toString());
		return lastTimeStepProbabilities;
	}

	public List<GraphiteQuery<JsonNode>> getPositiveQueries() {
		return positiveQueries;
	}

	public GraphiteBooleanClassification addPositiveQuery(GraphiteQuery<JsonNode> positiveQuery) {
		this.positiveQueries.add(positiveQuery);
		return this;
	}

	public List<GraphiteQuery<JsonNode>> getNegativeQueries() {
		return negativeQueries;
	}

	public GraphiteBooleanClassification addNegativeQuery(GraphiteQuery<JsonNode> negativeQuery) {
		this.negativeQueries.add(negativeQuery);
		return this;
	}

	public File getWorkingDirectory() {
		return workingDirectory;
	}

	public GraphiteBooleanClassification setWorkingDirectory(File workingDirectory) {
		this.workingDirectory = workingDirectory;
		return this;
	}

	public GraphiteClient getClient() {
		return client;
	}

	public GraphiteBooleanClassification setClient(QueryableGraphiteClient client) {
		this.client = client;
		return this;
	}

	public int getNumberOfIterations() {
		return numberOfIterations;
	}

	public GraphiteBooleanClassification setNumberOfIterations(int numberOfIterations) {
		this.numberOfIterations = numberOfIterations;
		return this;
	}

}
