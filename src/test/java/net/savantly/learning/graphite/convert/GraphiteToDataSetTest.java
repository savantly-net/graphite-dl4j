package net.savantly.learning.graphite.convert;

import java.io.IOException;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.List;

import org.assertj.core.util.Arrays;
import org.datavec.api.writable.Writable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.test.context.junit4.SpringRunner;

import com.fasterxml.jackson.databind.JsonNode;

import net.savantly.graphite.GraphiteClientFactory;
import net.savantly.graphite.QueryableGraphiteClient;
import net.savantly.graphite.query.GraphiteQueryBuilder;
import net.savantly.learning.graphite.domain.GraphiteMultiSeries;

@RunWith(SpringRunner.class)
public class GraphiteToDataSetTest {
private static Logger log = LoggerFactory.getLogger(GraphiteToDataSetTest.class);
	
	@Value("#{systemEnvironment['GRAPHITE_HOST']}")
	String graphiteHost;
	
	@Test
	public void testDataSet() throws UnknownHostException, SocketException, IOException {
		QueryableGraphiteClient client = GraphiteClientFactory.queryableGraphiteClient(graphiteHost);
		JsonNode jsonNode = client.query(GraphiteQueryBuilder.simpleQuery("alias(randomWalk(1), 'A')"));
		GraphiteMultiSeries series = GraphiteMultiSeries.from(jsonNode);
		DataSet dataSet = GraphiteToDataSet.toDataSet(123, series);
		log.info("{}",dataSet);
	}

	@Test
	public void testNDArray() throws UnknownHostException, SocketException, IOException {
		QueryableGraphiteClient client = GraphiteClientFactory.queryableGraphiteClient(graphiteHost);
		
		JsonNode jsonNode1 = client.query(GraphiteQueryBuilder.simpleQuery("alias(randomWalk(1), 'A')"));
		GraphiteMultiSeries series1 = GraphiteMultiSeries.from(jsonNode1);

		JsonNode jsonNode2 = client.query(GraphiteQueryBuilder.simpleQuery("alias(randomWalk(1), 'B')"));
		GraphiteMultiSeries series2 = GraphiteMultiSeries.from(jsonNode2);
		
		Pair<INDArray, INDArray> data = GraphiteToDataSet.toTimeSeriesNDArray(
				Arrays.array(Pair.of(123, series1), Pair.of(234, series2)));
		log.info("{}",data);
	}
	
	@Test
	public void testWritableSequence() throws UnknownHostException, SocketException, IOException {
		QueryableGraphiteClient client = GraphiteClientFactory.queryableGraphiteClient(graphiteHost);
		
		JsonNode jsonNode1 = client.query(GraphiteQueryBuilder.simpleQuery("alias(randomWalk(1), 'A')"));
		GraphiteMultiSeries series1 = GraphiteMultiSeries.from(jsonNode1);

		JsonNode jsonNode2 = client.query(GraphiteQueryBuilder.simpleQuery("alias(randomWalk(1), 'B')"));
		GraphiteMultiSeries series2 = GraphiteMultiSeries.from(jsonNode2);
		
		Pair<INDArray, INDArray> data = GraphiteToDataSet.toTimeSeriesNDArray(
				Arrays.array(Pair.of(123, series1), Pair.of(234, series2)));
		
		List<List<List<Writable>>> sequence = GraphiteToDataSet.toWritableSequence(data.getLeft());
		
		log.info("{}",sequence);
	}
}
