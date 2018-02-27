package net.savantly.learning.graphite.convert;

import java.io.IOException;
import java.net.SocketException;
import java.net.UnknownHostException;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.nd4j.linalg.dataset.api.DataSet;
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
	public void test() throws UnknownHostException, SocketException, IOException {
		QueryableGraphiteClient client = GraphiteClientFactory.queryableGraphiteClient(graphiteHost);
		JsonNode jsonNode = client.query(GraphiteQueryBuilder.simpleQuery("alias(randomWalk(1), 'A')"));
		GraphiteMultiSeries series = GraphiteMultiSeries.from(jsonNode);
		DataSet dataSet = GraphiteToDataSet.toDataSet(123, series);
		log.info("{}",dataSet);
	}

}
